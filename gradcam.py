# corrected_gradcam.py
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import copy
import os
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def get_model(num_classes):
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def get_preprocessing_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

class Denormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std = torch.tensor(std).view(3,1,1)
    def __call__(self, tensor):
        return tensor * self.std + self.mean

if __name__ == '__main__':
    MODEL_PATH = 'fl_dp_encrypted_final.pth'   # update if needed
    DATA_DIR = 'dataset_root'
    NUM_IMAGES = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    full_dataset = datasets.ImageFolder(DATA_DIR)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    torch.manual_seed(42)
    test_split = 0.2
    test_size = int(len(full_dataset) * test_split)
    train_size = len(full_dataset) - test_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])

    test_dataset.dataset = copy.deepcopy(test_dataset.dataset)
    test_dataset.dataset.transform = get_preprocessing_transforms()
    test_loader = DataLoader(test_dataset, batch_size=NUM_IMAGES, shuffle=True, num_workers=0)

    model = get_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])


    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    denormalize = Denormalize([0.485,0.456,0.406], [0.229,0.224,0.225])

    for i in range(min(NUM_IMAGES, inputs.size(0))):
        input_tensor = inputs[i:i+1]
        rgb_img_tensor = denormalize(input_tensor.squeeze(0).cpu()).clamp(0,1)
        rgb_img = np.float32(rgb_img_tensor.permute(1,2,0).numpy())

        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred_idx = torch.max(outputs, 1)

        targets = [ClassifierOutputTarget(pred_idx.item())]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        filename = f"gradcam_pred_{class_names[pred_idx.item()]}_true_{class_names[labels[i].item()]}_{i+1}.png"
        cv2.imwrite(filename, cam_image)

    print("Done — saved Grad-CAM images.")
