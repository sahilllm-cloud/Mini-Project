import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, models, transforms

# ---------------------------
# Model helper (ResNet18)
# ---------------------------
def get_model(num_classes, pretrained=True):
    # Use torchvision's modern API when available, fallback otherwise
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    except Exception:
        model = models.resnet18(pretrained=pretrained)
    # Freeze feature extractor by default; only train final fc
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ---------------------------
# Non-IID Dirichlet split
# ---------------------------
def create_non_iid_split(dataset, num_clients, alpha=0.5, min_size=10, max_attempts=10):
    """
    Returns a list of Subset objects (one per client).
    Works for a dataset or a random_split(SubSet) result.
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
                    selected = idxs[pointer:pointer + cnt]
                    client_idx_lists[client_id].extend(selected.tolist())
                    pointer += cnt
        sizes = [len(lst) for lst in client_idx_lists]
        if min(sizes) >= min_size:
            break
    else:
        print("Warning: couldn't satisfy min_size after attempts — some clients may be smaller than min_size.")

    client_subsets = []
    for lst in client_idx_lists:
        mapped = [int(global_indices[i]) for i in lst]
        subset = Subset(global_dataset, mapped)
        client_subsets.append(subset)

    for i, s in enumerate(client_subsets):
        print(f"Client {i}: {len(s)} samples")
    return client_subsets

# ---------------------------
# Client
# ---------------------------
class Client:
    def __init__(self, client_id, dataset, batch_size, num_classes, device, num_workers=0):
        self.client_id = client_id
        self.device = device
        # On Windows, set num_workers=0 if you hit dataloader worker issues
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=(device.type == 'cuda'))
        self.model = get_model(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                   lr=0.001, momentum=0.9)

    def set_global_model(self, global_state_dict):
        mapped = {k: v.to(self.device) for k, v in global_state_dict.items()}
        self.model.load_state_dict(mapped)

    def train(self, local_epochs=1):
        self.model.train()
        total_samples = 0
        for epoch in range(local_epochs):
            epoch_samples = 0
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_samples += labels.size(0)
            total_samples += epoch_samples
            print(f"Client {self.client_id} finished epoch {epoch+1}/{local_epochs} (processed {epoch_samples} samples).")
        print(f"Client {self.client_id} finished training: {total_samples} samples in {local_epochs} local epochs.")

    def get_local_weights(self):
        # Always return CPU tensors to avoid device mismatch during aggregation
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

# ---------------------------
# Server
# ---------------------------
class Server:
    def __init__(self, global_model):
        # Keep server model on CPU
        self.global_model = global_model.to('cpu')

    def aggregate_weights(self, client_weights_list):
        """
        Robust FedAvg aggregation:
        - ensures all client tensors are CPU
        - checks key consistency and shapes
        - sums in float64 on CPU and averages
        """
        if len(client_weights_list) == 0:
            print("aggregate_weights: no client weights provided.")
            return

        # Ensure all client weights are CPU tensors and keys are consistent
        client_weights_cpu = []
        key_sets = []
        for idx, cw in enumerate(client_weights_list):
            cw_cpu = {}
            if not isinstance(cw, dict):
                raise TypeError(f"Client weights at index {idx} is not a dict.")
            for k, v in cw.items():
                if not isinstance(v, torch.Tensor):
                    try:
                        v = torch.tensor(v)
                    except Exception as e:
                        raise TypeError(f"Could not convert client weight value for key '{k}' to tensor: {e}")
                cw_cpu[k] = v.detach().cpu()
            client_weights_cpu.append(cw_cpu)
            key_sets.append(set(cw_cpu.keys()))

        # Check that all clients share the same keys
        first_keys = key_sets[0]
        for i, ks in enumerate(key_sets[1:], start=1):
            if ks != first_keys:
                missing_in_i = first_keys - ks
                extra_in_i = ks - first_keys
                raise RuntimeError(f"Client {i} has mismatching parameter keys. Missing: {missing_in_i}, Extra: {extra_in_i}")

        # Reference global state dict (server model should be on CPU)
        global_state = self.global_model.state_dict()
        global_keys = set(global_state.keys())
        if global_keys != first_keys:
            missing = global_keys - first_keys
            extra = first_keys - global_keys
            raise RuntimeError(f"Global model keys mismatch with client keys. Missing in clients: {missing}, Extra in clients: {extra}")

        # Create CPU float64 accumulators with same shapes
        accum = {}
        for k, v in global_state.items():
            accum[k] = torch.zeros_like(v, dtype=torch.float64, device='cpu')

        # Sum client weights into accum; also validate shapes
        for idx, cw in enumerate(client_weights_cpu):
            for k in accum.keys():
                client_tensor = cw[k]
                if client_tensor.shape != accum[k].shape:
                    raise RuntimeError(f"Shape mismatch for key '{k}' from client {idx}: client shape {client_tensor.shape} vs server shape {accum[k].shape}")
                accum[k] += client_tensor.to(torch.float64)

        # Average and cast back to server dtype
        n_clients = len(client_weights_cpu)
        new_state = {}
        for k, agg in accum.items():
            new_state[k] = (agg / n_clients).to(global_state[k].dtype)

        # Load new state to server model (server.global_model is on CPU)
        self.global_model.load_state_dict(new_state)
        print(f"Server aggregated global model (FedAvg) from {n_clients} clients.")

    def get_global_model_state(self):
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

# ---------------------------
# Evaluation
# ---------------------------
def evaluate_model(model, dataloader, device):
    model = model.to(device)
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    return (running_corrects.double() / total).item() if total > 0 else 0.0

# ---------------------------
# Main script
# ---------------------------
if __name__ == "__main__":
    # ---------------------------
    # Config (tweak as needed)
    # ---------------------------
    NUM_CLIENTS = 3
    NUM_ROUNDS = 5
    LOCAL_EPOCHS = 10
    DIRICHLET_ALPHA = 0.5   # increase (e.g., 5.0) for more balanced splits
    DATA_DIR = 'dataset_root'  # ImageFolder root
    BATCH_SIZE = 32
    NUM_WORKERS = 0         # set to 0 on Windows if you face dataloader issues

    print("--- Phase 2: Federated Learning (No Privacy) ---")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # Dataset loading
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms['train'])
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images belonging to {num_classes} classes.")

    torch.manual_seed(42)
    test_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    # Ensure test_dataset uses validation transforms
    test_dataset.dataset = copy.deepcopy(test_dataset.dataset)
    test_dataset.dataset.transform = data_transforms['val']
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=(device.type == 'cuda'))

    # Create non-iid splits for clients
    client_subsets = create_non_iid_split(train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA, min_size=20)
    clients = [Client(i, client_subsets[i], batch_size=BATCH_SIZE, num_classes=num_classes,
                      device=device, num_workers=NUM_WORKERS) for i in range(NUM_CLIENTS)]

    # Initialize server + global model
    global_model = get_model(num_classes).to('cpu')
    server = Server(global_model)

    # Optionally load centralized model as initialization
    centralized_path = 'centralized_model.pth'
    if os.path.exists(centralized_path):
        print("Loading centralized model weights as initial global model...")
        state = torch.load(centralized_path, map_location='cpu')
        server.global_model.load_state_dict(state)

    # FL loop
    round_nums = []
    global_accs = []
    start_time = time.time()

    for round_idx in range(NUM_ROUNDS):
        print(f"\n--- Round {round_idx+1}/{NUM_ROUNDS} ---")
        client_weights_list = []
        global_state = server.get_global_model_state()

        # Each client receives global model, trains locally
        for client in clients:
            client.set_global_model(global_state)
            client.train(local_epochs=LOCAL_EPOCHS)
            client_weights_list.append(client.get_local_weights())

        # Aggregate on server (robust CPU aggregation)
        try:
            server.aggregate_weights(client_weights_list)
        except Exception as e:
            # Extra debug if something still goes wrong
            print("Aggregation failed with exception:", e)
            for i, cw in enumerate(client_weights_list):
                sample_key = next(iter(cw.keys()))
                print(f"Client {i} sample tensor device for key '{sample_key}': {cw[sample_key].device}, shape: {cw[sample_key].shape}")
            raise

        # Evaluate
        accuracy = evaluate_model(server.global_model, test_loader, device)
        round_nums.append(round_idx + 1)
        global_accs.append(accuracy)
        print(f"Round {round_idx+1} Global Accuracy: {accuracy:.4f}")

        # Save checkpoint for this round
        ckpt_path = f'global_round_{round_idx+1}.pth'
        torch.save(server.global_model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    total_time = time.time() - start_time
    print(f"\nFL Training Complete in {int(total_time // 60)}m {int(total_time % 60)}s")
    # Final save
    final_path = 'global_model_final.pth'
    torch.save(server.global_model.state_dict(), final_path)
    print(f"Saved final global model: {final_path}")
