import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

print("--- Task 3: Final Model Evaluation ---")

# --- 1. Define Model Architecture (match training) ---
def get_model(num_classes):
    # We load saved weights next, so starting from random or pretrained doesn't matter for loading.
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# --- 2. Val/Test transforms (match training 'val') ---
def get_test_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# --- 3. Prediction helper ---
def get_predictions(model, dataloader, device):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)

# --- 4. Main ---
if __name__ == '__main__':
    # ---- Config ----
    DATA_DIR = 'dataset_root'
    model_paths = {
    "Centralized": "centralized_model.pth",
    "Federated (No DP)": "global_model_final.pth",   # <-- use this
    "Federated (DP + Enc)": "fl_dp_encrypted_final.pth"
}


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Data ----
    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found classes: {class_names}")

    torch.manual_seed(42)
    test_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - test_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])

    test_dataset.dataset = copy.deepcopy(test_dataset.dataset)
    test_dataset.dataset.transform = get_test_transform()

    # Windows-friendly: num_workers=0
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"Test set loaded with {len(test_dataset)} images.")

    # ---- Evaluate models + plot macro ROC ----
    results = {}
    plt.figure(figsize=(10, 8))

    for model_name, model_path in model_paths.items():
        if not os.path.exists(model_path):
            print(f"Warning: model not found -> '{model_path}'. Skipping {model_name}.")
            continue

        print(f"\n--- Evaluating: {model_name} ---")
        model = get_model(num_classes)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)

        y_true, y_probs = get_predictions(model, test_loader, device)
        y_pred = np.argmax(y_probs, axis=1)

        # Classification report
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        results[model_name] = {
            'Accuracy': report['accuracy'],
            'Precision (macro)': report['macro avg']['precision'],
            'Recall (macro)': report['macro avg']['recall'],
            'F1-score (macro)': report['macro avg']['f1-score'],
        }

        # ROC (macro-average). Handle classes with no positives in test split.
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        fpr, tpr, roc_auc = {}, {}, {}
        valid_class_count = 0

        for i in range(num_classes):
            # If a class never appears as positive, roc_curve will error; skip it.
            if y_true_bin[:, i].sum() == 0 or y_true_bin[:, i].sum() == y_true_bin.shape[0]:
                continue
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            valid_class_count += 1

        if valid_class_count == 0:
            print("Skipping ROC curve (no valid positive classes in test split).")
            continue

        # Macro-average ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in fpr.keys()]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in fpr.keys():
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(fpr)

        fpr_macro = all_fpr
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)

        plt.plot(fpr_macro, tpr_macro, label=f'{model_name} (Macro AUC = {roc_auc_macro:.3f})')

    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (Macro-Average) — Centralized vs Federated')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('final_roc_comparison.png', dpi=150)
    plt.show()
    print("\nSaved ROC comparison plot as 'final_roc_comparison.png'")

    # ---- Print table ----
    print("\n--- Final Performance Comparison ---")
    print(f"{'Model':<25} | {'Accuracy':<8} | {'Precision':<10} | {'Recall':<8} | {'F1-Score':<9}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<25} | {metrics['Accuracy']:.4f}  | {metrics['Precision (macro)']:.4f}     | "
              f"{metrics['Recall (macro)']:.4f}  | {metrics['F1-score (macro)']:.4f}")

