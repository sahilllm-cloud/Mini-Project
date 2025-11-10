import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import os
import time
import copy

# --- Training and Evaluation Function ---
def train_model(model, criterion, optimizer, scheduler, num_epochs=100):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test':
                scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'🌟 New best test accuracy: {best_acc:.4f}')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'✅ Best Test Accuracy: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Phase 1: Centralized Training ---")

    # 1. Device Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Augmentation and Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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

    # 3. Load Dataset
    data_dir = 'dataset_root'
    full_dataset = datasets.ImageFolder(data_dir, data_transforms['train'])
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found classes: {class_names} (Total: {num_classes})")

    # 4. Train-Test Split
    torch.manual_seed(42)
    test_split = 0.2
    test_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_dataset.dataset.transform = data_transforms['val']

    # 5. Dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    }
    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}

    # 6. Model Setup (ResNet18 Fine-tuning)
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Fine-tune last few layers + final FC
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)
    print("✅ Model ready (ResNet18 fine-tuned).")

    # 7. Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 8. Train the Model
    print("🚀 Starting centralized training on GPU...")
    central_model = train_model(model, criterion, optimizer, scheduler, num_epochs=100)

    # 9. Save Model
    model_save_path = 'centralized_model.pth'
    torch.save(central_model.state_dict(), model_save_path)
    print(f"💾 Centralized model saved to: {model_save_path}")
    print("--- Phase 1 Complete ---")
