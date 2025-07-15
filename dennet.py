import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch 
import torch.nn as nn
import torchvision.models as models
from torchvision.models import densenet201, DenseNet201_Weights
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchcam.methods import GradCAM, SmoothGradCAMpp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import streamlit as st
from io import BytesIO
import cv2 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import random


# Function to set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For CUDA
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures repeatability

set_seed(42)  # Call it early in your code


train_dir = 'C:/Users/USER/Downloads/Nigeria Chest X-ray Dataset/my_dataset/train_folder'
test_dir = 'C:/Users/USER/Downloads/Nigeria Chest X-ray Dataset/my_dataset/test_folder'
classes = ['TB','PNEUMONIA','NORMAL','COVID']
# To count the images per classes
train_count = {cls:len(os.listdir(os.path.join(train_dir,cls))) for cls in classes}
df_counts = pd.DataFrame.from_dict(train_count, orient ='index', columns =['count'])
# Plotting the class distribution
sns.barplot(x=df_counts.index, y = df_counts['count'])
plt.title('Class Dirtribution in Training Set')
plt.xticks(rotation=45)
plt.show()
# To view sample class images
fig, axes = plt.subplots(1,4, figsize=(15,3))
for i, cls in enumerate(classes):
    img_path =  os.path.join(train_dir, cls, os.listdir(os.path.join(train_dir, cls))[0])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    axes[i].imshow(img, cmap = 'gray')
    axes[i].set_title(cls)
    axes[i].axis('off')
plt.show()
# Creating a custom data to apply albumentations
class XRayDataset(Dataset):
    def __init__(self, root_dir, transform = None, max_images_per_class = 300):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['TB','PNEUMONIA','NORMAL','COVID']
        self.image_paths = []
        self.labels = []
        print('Loading dataset...')
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            paths = [os.path.join(root_dir, cls, f) for f in os.listdir(cls_dir)if  f.endswith('.png')]
            paths = paths[:max_images_per_class]
            self.image_paths.extend(paths)
            self.labels.extend([idx] * len(paths))
            print(f'Loaded {len(self.image_paths)}images')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f'Failed to load image {img_path}')
            raise ValueError(f'Image at {img_path} not loaded')
        if self.transform:
            augumented = self.transform(image = img)
            img = augumented['image']
        return img, label
# Defining transforms
train_transform = A.Compose([
    A.Resize(128,128),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.Normalize(mean=0.5, std=0.5),
    #A.set_seed(42),  # Set seed for reproducibility
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(128,128),
    A.Normalize(mean=0.5, std=0.5),
    ToTensorV2()
])

# Creating dataset and loaders
print('Creating data loaders...')
train_dataset = XRayDataset(train_dir, transform = train_transform, max_images_per_class = 300)
test_dataset = XRayDataset(test_dir, transform = test_transform, max_images_per_class = 300)
train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 0)
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 0)
print('Data Loader Created')

# Compute class weight
class_counts = [max(1,len(os.listdir(os.path.join(train_dir, cls)))) for cls in classes]
total_count = sum(class_counts)
class_weights = torch.tensor([total_count/(len(classes)*count)for count in class_counts],dtype=torch.float)
print(f'Class Count: {class_counts}')
print(f'Class Weight: {class_weights}')

# Defining the model
model = models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
old_weights = model.features.conv0.weight.data
model.conv1 = nn.Conv2d(1, 64, kernel_size = 7, stride= 2, padding = 3, bias=False)
# Copy the old weights to the new conv1 layer
#model.conv1.weight.data = old_weights.mean(dim=1, keepdim=True)  # Average the channels to convert from 3 to 1 channel
model.features.conv0.weight.data = old_weights.mean(dim=1, keepdim=True)
num_features = model.classifier.in_features
model.classifier = nn.Linear(model.classifier.in_features, len(classes))

# If GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
class_weights = class_weights.to(device)
print(f'Using Device:{device}')

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
print(model(torch.randn(1, 1, 128, 128)).shape)


# Creating a training loop
num_epochs = 20 # I can chose any number I want, but 3 will make the training faster
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
print('Starting training....')
for epoch in range (num_epochs):
    model.train()
    running_loss = 0.0
    for images, label in train_loader:
        images, label = images.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 5 == 4:
            #print(f'Epoch{epoch+1}, Loss:{loss.item()}:.4f')
            print(f'Epoch{epoch+1}, Batch{i+1}, Loss:{running_loss/5:.4f}')
            running_loss =0.0
    scheduler.step()  # Step the learning rate scheduler
print('Training Complete')

# Saving the model
torch.save(model.state_dict(), 'dennet.pth')
print('Model saved as dennet_model.pth')
#model_path = 'xray_model.pth'
    
# Evaluating the model
model.eval()
print('Evaluating Model....')
y_true, y_pred = [],[]
with torch.no_grad():
    #for images, label in test_loader:
    for i, (images, label) in enumerate(test_loader):
        print(f'Evaluating Batch{i+1}/{len(test_loader)}...')
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs,1)
        y_true.extend(label.numpy())
        y_pred.extend(preds.cpu().numpy())
print('Evaluation Complete')

# Classification Report
print('\nClassification Report:')
print(classification_report(y_true,y_pred,target_names = classes))

# Accuracy Score
print('\nAccuracy Score:')
print(round(accuracy_score(y_true, y_pred)*100,2))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Comfusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Initialize Grad-CAM with the target layer (layer4 for DenseNet)
cam = GradCAM(model, target_layer='features.denseblock4')

# Load and preprocess the image
img_path = 'C:/Users/USER/Downloads/Nigeria Chest X-ray Dataset/my_dataset/train_folder/COVID/COVID-6.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError(f"Failed to load image: {img_path}")

# Apply test transform (same as used in your dataset)
img_tensor = test_transform(image=img)['image'].unsqueeze(0).to(device)  # Shape: [1, 1, 128, 128]

# Forward pass to get model output and hook features
img_tensor.requires_grad_(True)  # Enable gradients for the input tensor
scores = model(img_tensor)  # Forward pass with gradients enabled

# Get the predicted class index
class_idx = torch.argmax(scores, dim=1).item()
print(f"Predicted class: {classes[class_idx]} (index: {class_idx})")

# Compute Grad-CAM
cam_map = cam(class_idx=class_idx, scores=scores)[0]  # Get CAM for the predicted class
cam_map = cam_map.cpu().numpy()

# Resize CAM to match original image size
cam_map = cv2.resize(cam_map, (img.shape[1], img.shape[0]))

# Normalize CAM for visualization
cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min())

# Create heatmap and overlay
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original X-ray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.imshow(cam_map, cmap='jet', alpha=0.5)  # Overlay heatmap
plt.title(f'Grad-CAM: {classes[class_idx]}')
plt.axis('off')
plt.show()

