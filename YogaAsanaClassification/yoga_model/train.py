import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import sys

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from yoga_model.model import YogaCNN

# Dataset definition
class YogaDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

        # Extract asana and quality class names from the folder names
        self.asana_classes = sorted(list({os.path.basename(os.path.dirname(p)).split('_')[0] for p in image_paths}))
        self.quality_classes = ['Poor', 'Average', 'Good']

        self.asana_to_idx = {cls: idx for idx, cls in enumerate(self.asana_classes)}
        self.quality_to_idx = {q: idx for idx, q in enumerate(self.quality_classes)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        parts = os.path.basename(os.path.dirname(img_path)).split('_')
        asana_label = self.asana_to_idx[parts[0]]
        quality_label = self.quality_to_idx[parts[1]] if len(parts) > 1 else 1  # Default to 'Average'

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(asana_label), torch.tensor(quality_label)

# Helper to validate image
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset directories
train_dir = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\final_dataset\train"
val_dir = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\final_dataset\val"

# Collect valid image paths
def get_image_paths(directory):
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files if is_valid_image(os.path.join(root, file))
    ]

train_paths = get_image_paths(train_dir)
val_paths = get_image_paths(val_dir)

# Dataset and DataLoader
train_dataset = YogaDataset(train_paths, transform=transform)
val_dataset = YogaDataset(val_paths, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model, loss, optimizer
model = YogaCNN(num_classes=len(train_dataset.asana_to_idx), quality_classes=3)
criterion_asana = nn.CrossEntropyLoss()
criterion_quality = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, asana_labels, quality_labels in train_loader:
        optimizer.zero_grad()
        asana_outputs, quality_outputs = model(images)
        loss_asana = criterion_asana(asana_outputs, asana_labels)
        loss_quality = criterion_quality(quality_outputs, quality_labels)
        loss = loss_asana + loss_quality
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
