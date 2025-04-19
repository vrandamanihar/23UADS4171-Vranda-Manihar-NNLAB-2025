import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from yoga_model.model import YogaCNN
from yoga_model.train import YogaDataset  # reuse dataset class

# Define image validator
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except:
        return False

# Load test image paths
test_dir = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\final_dataset\test"
test_paths = [
    os.path.join(root, file)
    for root, _, files in os.walk(test_dir)
    for file in files if is_valid_image(os.path.join(root, file))
]

# Define transforms
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create dataset and loader
test_dataset = YogaDataset(test_paths, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=16)

# Load model
model = YogaCNN(num_classes=len(test_dataset.asana_to_idx), quality_classes=3)
model.load_state_dict(torch.load("yoga_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Evaluate
correct_asana = 0
correct_quality = 0
total = 0

with torch.no_grad():
    for images, asana_labels, quality_labels in test_loader:
        asana_outputs, quality_outputs = model(images)
        _, asana_preds = torch.max(asana_outputs, 1)
        _, quality_preds = torch.max(quality_outputs, 1)

        total += asana_labels.size(0)
        correct_asana += (asana_preds == asana_labels).sum().item()
        correct_quality += (quality_preds == quality_labels).sum().item()

print(f"Asana Accuracy: {100 * correct_asana / total:.2f}%")
print(f"Quality Accuracy: {100 * correct_quality / total:.2f}%")
