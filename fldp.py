# federated_dp_encrypted.py
import os
import copy
import time
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from cryptography.fernet import Fernet
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms
from opacus import PrivacyEngine

# ---------------------------
# Helpers
# ---------------------------
def get_model(num_classes, pretrained=True):
    try:
        # modern torchvision API
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    except Exception:
        model = models.resnet18(pretrained=pretrained)
    # Freeze all layers except final fc
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_non_iid_subsets(dataset, num_clients, alpha=0.5, min_size=10, max_attempts=10):
    """
    Return list of Subset objects (one per client) using Dirichlet partitioning.
    """
    if isinstance(dataset, Subset):
        global_dataset = dataset.dataset
        global_indices = np.array(dataset.indices)
        targets = np.array([global_dataset.targets[i] for i in global_indices])
    else:
        global_dataset = dataset
        global_indices = np.arange(len(global_dataset))
        targets = np.array(global_dataset.targets)

    num_classes = int(np.max(targets)) + 1
    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]

    for attempt in range(max_attempts):
        client_idx_lists = [[] for _ in range(num_clients)]
        for c, idxs in enumerate(class_indices):
            if len(idxs) == 0:
                continue
            proportions = np.random.dirichlet([alpha] * num_clients)
            counts = (proportions * len(idxs)).astype(int)
            remainder = len(idxs) - counts.sum()
            if remainder > 0:
                for i in np.argsort(proportions)[-remainder:]:
                    counts[i] += 1
            np.random.shuffle(idxs)
            pointer = 0
            for client_id in range(num_clients):
                cnt = counts[client_id]
                if cnt > 0:
                    selected = idxs[pointer:pointer+cnt]
                    client_idx_lists[client_id].extend(selected.tolist())
                    pointer += cnt
        sizes = [len(lst) for lst in client_idx_lists]
        if min(sizes) >= min_size:
            break
    else:
        print("Warning: couldn't satisfy min_size after attempts; some clients may be small.")

    client_subsets = []
    for lst in client_idx_lists:
        mapped = [int(global_indices[i]) for i in lst]
        client_subsets.append(Subset(global_dataset, mapped))

    for i, s in enumerate(client_subsets):
        print(f"Client {i}: {len(s)} samples")
    return client_subsets

# ---------------------------
# Client class (hospital)
# ---------------------------
class Client:
    def __init__(self, client_id, subset_dataset, num_classes, device,
                 noise_multiplier=2, max_grad_norm=1.0, batch_size=32, num_workers=0):
        self.client_id = client_id
        self.device = device
        self.batch_size = batch_size
        self.subset = subset_dataset
        # DataLoader used by Opacus must be the same dataset object
        self.dataloader = DataLoader(self.subset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        # model: we will wrap with Opacus later
        self.model = get_model(num_classes).to(self.device)
        # Only train final layer parameters (we frozen earlier)
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.001, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        # privacy_engine will be created in make_private() below
        self.privacy_engine = None

    def make_private(self, sample_size, epochs):
        """
        Wrap model/optimizer/dataloader with Opacus PrivacyEngine.
        sample_size: number of samples in this client's dataset (used by Opacus)
        epochs: planned local epochs (used for accounting)
        """
        privacy_engine = PrivacyEngine()
        # use make_private wrapper which returns wrapped objects
        try:
            self.model, self.optimizer, self.dataloader = privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=False,  # typically False for DataLoader without replacement
            )
        except TypeError:
            # fallback to attach API for older/newer opacus versions
            privacy_engine = PrivacyEngine(
                module=self.model,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
                sample_rate=self.batch_size / float(sample_size) if sample_size > 0 else 0.0,
            )
            privacy_engine.attach(self.optimizer)
        self.privacy_engine = privacy_engine

    def set_global_model(self, global_state_dict):
        # load into model; if wrapped, Opacus returns DPModule or similar; we try safe approaches
        try:
            # many Opacus APIs return a wrapped module where the original module is accessible via ._module
            if hasattr(self.model, "_module"):
                self.model._module.load_state_dict(global_state_dict)
            else:
                self.model.load_state_dict(global_state_dict)
        except Exception:
            # last resort: load to CPU version and then to device
            tmp = {k: v.to('cpu') for k, v in global_state_dict.items()}
            if hasattr(self.model, "_module"):
                self.model._module.load_state_dict(tmp)
            else:
                self.model.load_state_dict(tmp)

    def train(self, local_epochs=1, device=None, delta=1e-5):
        """Train locally for local_epochs. Returns epsilon (if available) or None."""
        if device is None:
            device = self.device
        # Ensure model on device
        try:
            # underlying module may be at ._module
            base_model = self.model._module if hasattr(self.model, "_module") else self.model
        except Exception:
            base_model = self.model

        base_model.train()
        total_steps = 0
        for epoch in range(local_epochs):
            for inputs, labels in self.dataloader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_steps += 1

        # Try to get epsilon from privacy engine if available
        epsilon = None
        try:
            if hasattr(self.privacy_engine, "get_epsilon"):
                epsilon = float(self.privacy_engine.get_epsilon(delta=delta))
            elif hasattr(self.privacy_engine, "accountant") and hasattr(self.privacy_engine.accountant, "get_epsilon"):
                epsilon = float(self.privacy_engine.accountant.get_epsilon(delta=delta))
        except Exception:
            epsilon = None

        print(f"Client {self.client_id} finished local training ({local_epochs} epochs, {total_steps} steps). Epsilon: {epsilon}")
        return epsilon

    def get_encrypted_weights(self, cipher_suite):
        """Serialize and encrypt the underlying model weights (.state_dict of base model)."""
        base = self.model._module if hasattr(self.model, "_module") else self.model
        weights = base.state_dict()
        # ensure CPU tensors for compact serialization
        cpu_weights = {k: v.detach().cpu() for k, v in weights.items()}
        serialized = pickle.dumps(cpu_weights)
        encrypted = cipher_suite.encrypt(serialized)
        return encrypted

# ---------------------------
# Server / Aggregator class
# ---------------------------
class Server:
    def __init__(self, global_model, cipher_suite):
        self.global_model = global_model  # keep on CPU or device as you prefer
        self.cipher_suite = cipher_suite

    def aggregate_weights(self, encrypted_weights_list):
        """Decrypt client weights, average them, and load into global_model (FedAvg)."""
        client_weights = []
        for enc in encrypted_weights_list:
            decrypted = self.cipher_suite.decrypt(enc)
            weights = pickle.loads(decrypted)
            client_weights.append(weights)

        if len(client_weights) == 0:
            return

        # start accumulators on CPU float64
        global_state = self.global_model.state_dict()
        accum = {k: torch.zeros_like(v, dtype=torch.float64, device='cpu') for k, v in global_state.items()}

        # sum
        for cw in client_weights:
            # ensure all keys exist
            for k in global_state.keys():
                if k not in cw:
                    raise RuntimeError(f"Missing key {k} in client weights")
                accum[k] += cw[k].to(torch.float64)

        # average and add to global (here clients send absolute weights; FedAvg is average of weights)
        n = len(client_weights)
        new_state = {}
        for k in global_state.keys():
            avg = (accum[k] / n).to(global_state[k].dtype)
            new_state[k] = avg

        # Load averaged weights into global model
        self.global_model.load_state_dict(new_state)
        print(f"Server aggregated global model (FedAvg) from {n} clients.")

    def get_global_model_state(self):
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

# ---------------------------
# Evaluation (same as Phase 2)
# ---------------------------
def evaluate_model(model, dataloader, device):
    model = model.to(device)
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    return (running_corrects.double() / total).item() if total > 0 else 0.0

# ---------------------------
# Main execution
# ---------------------------
if __name__ == "__main__":
    # Config
    NUM_CLIENTS = 3
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 3
    DIRICHLET_ALPHA = 0.5
    DATA_DIR = 'dataset_root'
    BATCH_SIZE = 32
    NUM_WORKERS = 0  # set 0 on Windows

    # DP / Opacus params
    NOISE_MULTIPLIER = 2
    MAX_GRAD_NORM = 1.0
    DELTA = 1e-5

    print("Phase 3: FL + Opacus DP + Encrypted transfer")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Encryption key
    key = Fernet.generate_key()
    cipher = Fernet(key)
    print("Generated Fernet key (keep secret):", key.decode())

    # Data & transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    }

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images across {num_classes} classes.")

    torch.manual_seed(42)
    test_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    # set test transform
    test_dataset.dataset = copy.deepcopy(test_dataset.dataset)
    test_dataset.dataset.transform = data_transforms['val']
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Create client subsets
    client_subsets = create_non_iid_subsets(train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA, min_size=20)
    # Instantiate clients (give each its subset and let client create dataloader)
    clients = []
    for i, subset in enumerate(client_subsets):
        c = Client(client_id=i, subset_dataset=subset, num_classes=num_classes, device=device,
                   noise_multiplier=NOISE_MULTIPLIER, max_grad_norm=MAX_GRAD_NORM,
                   batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        # Prepare privacy wrapper (pass sample_size=len(subset) and planned epochs)
        c.make_private(sample_size=len(subset), epochs=LOCAL_EPOCHS)
        clients.append(c)

    # Server and global model (keep global model on CPU for aggregation)
    global_model = get_model(num_classes).to('cpu')
    server = Server(global_model, cipher)

    # Optionally initialize from centralized checkpoint if present
    if os.path.exists('centralized_model.pth'):
        state = torch.load('centralized_model.pth', map_location='cpu')
        server.global_model.load_state_dict(state)
        print("Loaded centralized model initialization.")

    # FL loop
    history_rounds = []
    history_acc = []
    history_eps = []

    start_time = time.time()
    for r in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n--- Round {r+1}/{NUM_ROUNDS} ---")
        encrypted_weights = []
        eps_sum = 0.0

        global_state = server.get_global_model_state()

        # Each client receives global, trains, sends encrypted weights
        for c in clients:
            c.set_global_model(global_state)
            eps = c.train(local_epochs=LOCAL_EPOCHS, device=device, delta=DELTA)
            if eps is not None:
                eps_sum += eps
            enc = c.get_encrypted_weights(cipher)
            encrypted_weights.append(enc)

            # Log to file (binary)
            with open("encrypted_transmission.log", "ab") as f:
                f.write(f"--- Round {r+1}, Client {c.client_id} ---\n".encode('utf-8'))
                f.write(enc + b"\n\n")

        # Server aggregates (decrypts + average)
        server.aggregate_weights(encrypted_weights)

        # Evaluate on test set (use GPU device for evaluation)
        acc = evaluate_model(server.global_model, test_loader, device)
        avg_eps = (eps_sum / len(clients)) if eps_sum > 0 else None

        history_rounds.append(r + 1)
        history_acc.append(acc)
        history_eps.append(avg_eps)

        round_time = time.time() - round_start
        print(f"Round {r+1} accuracy: {acc:.4f} | avg epsilon: {avg_eps} | time: {round_time:.1f}s")

        # Save checkpoint
        torch.save(server.global_model.state_dict(), f"global_round_{r+1}.pth")

    total_time = time.time() - start_time
    print(f"\nDP-FL finished in {total_time//60:.0f}m {total_time%60:.0f}s")
    torch.save(server.global_model.state_dict(), "fl_dp_encrypted_final.pth")
    print("Saved final model: fl_dp_encrypted_final.pth")
