# yoga_model/test.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os # Import os to check for checkpoint directory

# Corrected imports
from dataset import YogaDataset
from model import YogaCNN

# --- Configuration ---
# Match these with your training setup
ASANA_CLASSES = 6
RATING_CLASSES = 3
BATCH_SIZE = 16 # Use the same or a suitable batch size for testing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "yoga_cnn.pth"
TEST_DATA_DIR = "processed_data/final_dataset/test" # Make sure this path is correct relative to where you run test.py
# Or use an absolute path if running from different locations
# Example: TEST_DATA_DIR = r"D:\mini project NN\YogaAsanaClassification\processed_data\final_dataset\test"

# --- Checkpoint Path ---
checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint file not found at {checkpoint_path}")
    print("Please ensure the training script ran successfully and saved the model.")
    exit() # Exit if checkpoint doesn't exist

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Model Initialization ---
# Use the same arguments as during training
model = YogaCNN(asana_classes=ASANA_CLASSES, rating_classes=RATING_CLASSES)
try:
    # Load the state dict, mapping to the correct device
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    print("Ensure the model architecture in model.py matches the saved checkpoint.")
    exit()

model.to(device) # Move model to the appropriate device
model.eval() # Set model to evaluation mode

# --- Data Loading ---
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

try:
    test_dataset = YogaDataset(TEST_DATA_DIR, transform=test_transforms)
    if len(test_dataset) == 0:
        print(f"Error: No data found in the test directory: {TEST_DATA_DIR}")
        exit()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    print(f"Loaded {len(test_dataset)} images from {TEST_DATA_DIR}")
except FileNotFoundError:
    print(f"Error: Test data directory not found at {TEST_DATA_DIR}")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()


# --- Testing Loop ---
correct_asana = 0
correct_rating = 0
total = 0

with torch.no_grad(): # Disable gradient calculations for inference
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # Get model outputs (asana and rating)
        asana_out, rating_out = model(images)

        # --- Asana Accuracy ---
        _, predicted_asana = torch.max(asana_out.data, 1)
        # Derive asana labels from the combined label (assuming same logic as train.py)
        asana_labels = labels // RATING_CLASSES # Use RATING_CLASSES here
        correct_asana += (predicted_asana == asana_labels).sum().item()

        # --- Rating Accuracy (Optional) ---
        _, predicted_rating = torch.max(rating_out.data, 1)
        # Derive rating labels from the combined label
        rating_labels = labels % RATING_CLASSES # Use RATING_CLASSES here
        correct_rating += (predicted_rating == rating_labels).sum().item()

        total += labels.size(0)

# --- Print Results ---
if total > 0:
    asana_accuracy = 100 * correct_asana / total
    rating_accuracy = 100 * correct_rating / total
    print(f"Test Asana Accuracy: {asana_accuracy:.2f}% ({correct_asana}/{total})")
    print(f"Test Rating Accuracy: {rating_accuracy:.2f}% ({correct_rating}/{total})")
else:
    print("No samples were processed.")
