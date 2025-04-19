# yoga_model/train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import YogaDataset # Assuming dataset.py is in the same directory
from model import YogaCNN     # Assuming model.py is in the same directory
import torch.nn as nn
import torch.optim as optim
import os
import time # Optional: for timing epochs

# --- Configuration ---
ASANA_CLASSES = 6
RATING_CLASSES = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
TRAIN_DATA_DIR =r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\final_dataset\train"
VAL_DATA_DIR =r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\final_dataset\val"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "yoga_cnn.pth"
# Consider adding num_workers based on your CPU # yoga_model/train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import YogaDataset # Assuming dataset.py is in the same directory
from model import YogaCNN     # Assuming model.py is in the same director 
import torch.nn as nn
import torch.optim as optim
import os
import time # Optional: for timing epochs

# --- Configuration ---
ASANA_CLASSES = 6
RATING_CLASSES = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50
TRAIN_DATA_DIR = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\yoga_model\processed_data\final_dataset\train"
VAL_DATA_DIR = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\yoga_model\processed_data\final_dataset\val"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_NAME = "yoga_cnn.pth"
# Consider adding num_workers based on your CPU cores, e.g., 2 or 4
NUM_WORKERS = 0 # Start with 0, increase if data loading is a bottleneck

# --- Ensure Checkpoint Directory Exists ---
if not os.path.exists(CHECKPOINT_DIR):
    print(f"Creating checkpoint directory: {CHECKPOINT_DIR}")
    os.makedirs(CHECKPOINT_DIR)

# --- Data Transforms ---
# Consider adding normalization if beneficial for your model/data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Example: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Datasets and DataLoaders ---
print("Loading datasets...")
try:
    train_dataset = YogaDataset(TRAIN_DATA_DIR, transform=transform)
    val_dataset = YogaDataset(VAL_DATA_DIR, transform=transform)

    if len(train_dataset) == 0:
        print(f"Error: No training data found in {TRAIN_DATA_DIR}")
        exit()
    if len(val_dataset) == 0:
        # Decide if you want to exit or just warn
        print(f"Warning: No validation data found in {VAL_DATA_DIR}. Proceeding without validation.")
        # val_loader = None # Option to disable validation if no data

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # Only create val_loader if val_dataset is not empty
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) if len(val_dataset) > 0 else None

    print(f"Loaded {len(train_dataset)} training samples.")
    if val_loader:
        print(f"Loaded {len(val_dataset)} validation samples.")
    print(f"Detected classes: {train_dataset.class_names}") # Useful for debugging label mapping

    # --- Verify Label Mapping Logic ---
    # This assumes your folder names directly map to the combined class index
    num_expected_classes = ASANA_CLASSES * RATING_CLASSES
    if len(train_dataset.class_names) != num_expected_classes:
         print(f"Warning: Number of detected class folders ({len(train_dataset.class_names)}) "
               f"does not match ASANA_CLASSES * RATING_CLASSES ({num_expected_classes}).")
         print("Check your data folders and the ASANA_CLASSES/RATING_CLASSES constants.")
         # Consider exiting if this mismatch is critical
         # exit()

except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure data directories exist relative to the script.")
    print(f"Searched for Train: {os.path.abspath(TRAIN_DATA_DIR)}")
    print(f"Searched for Val: {os.path.abspath(VAL_DATA_DIR)}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()


# --- Model, Loss, Optimizer ---
print("Initializing model...")
model = YogaCNN(ASANA_CLASSES, RATING_CLASSES)
criterion_asana = nn.CrossEntropyLoss()
criterion_rating = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")


# --- Training Loop ---
print(f"Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    start_time = time.time() # Optional: time the epoch
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_asana_train = 0
    correct_rating_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Derive labels (ensure RATING_CLASSES is correct based on your data structure)
        # Example: if folders are 'asana1_good', 'asana1_bad', 'asana2_good', ...
        # and RATING_CLASSES = 2 (good/bad), then use labels // 2 and labels % 2
        asana_labels = labels // RATING_CLASSES
        rating_labels = labels % RATING_CLASSES

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        asana_out, rating_out = model(images)

        # Calculate loss
        loss_asana = criterion_asana(asana_out, asana_labels)
        loss_rating = criterion_rating(rating_out, rating_labels)
        # Combine losses (consider weighting them if one task is more important, e.g., loss = 0.7*loss_asana + 0.3*loss_rating)
        loss = loss_asana + loss_rating

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # --- Statistics ---
        running_loss += loss.item() * images.size(0) # Accumulate weighted loss

        # Calculate accuracy for the batch
        _, predicted_asana = torch.max(asana_out.data, 1)
        _, predicted_rating = torch.max(rating_out.data, 1)
        total_train += labels.size(0)
        correct_asana_train += (predicted_asana == asana_labels).sum().item()
        correct_rating_train += (predicted_rating == rating_labels).sum().item()

        # Optional: Print batch progress
        # if (i + 1) % 50 == 0: # Print every 50 batches
        #     print(f'  Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}')

    # --- Calculate Epoch Statistics ---
    epoch_loss = running_loss / len(train_dataset) # Average loss for the epoch
    epoch_asana_acc = 100 * correct_asana_train / total_train
    epoch_rating_acc = 100 * correct_rating_train / total_train
    epoch_time = time.time() - start_time # Optional

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_loss:.4f} | "
          f"Train Asana Acc: {epoch_asana_acc:.2f}% | "
          f"Train Rating Acc: {epoch_rating_acc:.2f}% | "
          f"Time: {epoch_time:.2f}s") # Optional

    # --- Validation Loop ---
    if val_loader:
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_asana_val = 0
        correct_rating_val = 0
        total_val = 0
        with torch.no_grad(): # Disable gradient calculations
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                asana_labels = labels // RATING_CLASSES
                rating_labels = labels % RATING_CLASSES

                asana_out, rating_out = model(images)
                loss_asana = criterion_asana(asana_out, asana_labels)
                loss_rating = criterion_rating(rating_out, rating_labels)
                loss = loss_asana + loss_rating

                val_loss += loss.item() * images.size(0)
                _, predicted_asana = torch.max(asana_out.data, 1)
                _, predicted_rating = torch.max(rating_out.data, 1)
                total_val += labels.size(0)
                correct_asana_val += (predicted_asana == asana_labels).sum().item()
                correct_rating_val += (predicted_rating == rating_labels).sum().item()

        # --- Calculate Validation Statistics ---
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_asana_acc = 100 * correct_asana_val / total_val
        epoch_val_rating_acc = 100 * correct_rating_val / total_val
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Asana Acc: {epoch_val_asana_acc:.2f}% | "
              f"Val Rating Acc: {epoch_val_rating_acc:.2f}%")
    else:
        print(f"Epoch {epoch+1}/{EPOCHS} | No validation data to evaluate.")


# --- Save the Final Model ---
checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
try:
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining finished.")
    print(f"Model saved successfully to {checkpoint_path}")
except Exception as e:
    print(f"\nTraining finished.")
    print(f"Error saving model: {e}")

NUM_WORKERS = 0 # Start with 0, increase if data loading is a bottleneck

# --- Ensure Checkpoint Directory Exists ---
if not os.path.exists(CHECKPOINT_DIR):
    print(f"Creating checkpoint directory: {CHECKPOINT_DIR}")
    os.makedirs(CHECKPOINT_DIR)

# --- Data Transforms ---
# Consider adding normalization if beneficial for your model/data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Example: transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Datasets and DataLoaders ---
print("Loading datasets...")
try:
    train_dataset = YogaDataset(TRAIN_DATA_DIR, transform=transform)
    val_dataset = YogaDataset(VAL_DATA_DIR, transform=transform)

    if len(train_dataset) == 0:
        print(f"Error: No training data found in {TRAIN_DATA_DIR}")
        exit()
    if len(val_dataset) == 0:
        # Decide if you want to exit or just warn
        print(f"Warning: No validation data found in {VAL_DATA_DIR}. Proceeding without validation.")
        # val_loader = None # Option to disable validation if no data

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # Only create val_loader if val_dataset is not empty
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS) if len(val_dataset) > 0 else None

    print(f"Loaded {len(train_dataset)} training samples.")
    if val_loader:
        print(f"Loaded {len(val_dataset)} validation samples.")
    print(f"Detected classes: {train_dataset.class_names}") # Useful for debugging label mapping

    # --- Verify Label Mapping Logic ---
    # This assumes your folder names directly map to the combined class index
    num_expected_classes = ASANA_CLASSES * RATING_CLASSES
    if len(train_dataset.class_names) != num_expected_classes:
         print(f"Warning: Number of detected class folders ({len(train_dataset.class_names)}) "
               f"does not match ASANA_CLASSES * RATING_CLASSES ({num_expected_classes}).")
         print("Check your data folders and the ASANA_CLASSES/RATING_CLASSES constants.")
         # Consider exiting if this mismatch is critical
         # exit()

except FileNotFoundError as e:
    print(f"Error loading data: {e}. Make sure data directories exist relative to the script.")
    print(f"Searched for Train: {os.path.abspath(TRAIN_DATA_DIR)}")
    print(f"Searched for Val: {os.path.abspath(VAL_DATA_DIR)}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()


# --- Model, Loss, Optimizer ---
print("Initializing model...")
model = YogaCNN(ASANA_CLASSES, RATING_CLASSES)
criterion_asana = nn.CrossEntropyLoss()
criterion_rating = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")


# --- Training Loop ---
print(f"Starting training for {EPOCHS} epochs...")
for epoch in range(EPOCHS):
    start_time = time.time() # Optional: time the epoch
    model.train() # Set model to training mode
    running_loss = 0.0
    correct_asana_train = 0
    correct_rating_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Derive labels (ensure RATING_CLASSES is correct based on your data structure)
        # Example: if folders are 'asana1_good', 'asana1_bad', 'asana2_good', ...
        # and RATING_CLASSES = 2 (good/bad), then use labels // 2 and labels % 2
        asana_labels = labels // RATING_CLASSES
        rating_labels = labels % RATING_CLASSES

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        asana_out, rating_out = model(images)

        # Calculate loss
        loss_asana = criterion_asana(asana_out, asana_labels)
        loss_rating = criterion_rating(rating_out, rating_labels)
        # Combine losses (consider weighting them if one task is more important, e.g., loss = 0.7*loss_asana + 0.3*loss_rating)
        loss = loss_asana + loss_rating

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # --- Statistics ---
        running_loss += loss.item() * images.size(0) # Accumulate weighted loss

        # Calculate accuracy for the batch
        _, predicted_asana = torch.max(asana_out.data, 1)
        _, predicted_rating = torch.max(rating_out.data, 1)
        total_train += labels.size(0)
        correct_asana_train += (predicted_asana == asana_labels).sum().item()
        correct_rating_train += (predicted_rating == rating_labels).sum().item()

        # Optional: Print batch progress
        # if (i + 1) % 50 == 0: # Print every 50 batches
        #     print(f'  Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}')

    # --- Calculate Epoch Statistics ---
    epoch_loss = running_loss / len(train_dataset) # Average loss for the epoch
    epoch_asana_acc = 100 * correct_asana_train / total_train
    epoch_rating_acc = 100 * correct_rating_train / total_train
    epoch_time = time.time() - start_time # Optional

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {epoch_loss:.4f} | "
          f"Train Asana Acc: {epoch_asana_acc:.2f}% | "
          f"Train Rating Acc: {epoch_rating_acc:.2f}% | "
          f"Time: {epoch_time:.2f}s") # Optional

    # --- Validation Loop ---
    if val_loader:
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct_asana_val = 0
        correct_rating_val = 0
        total_val = 0
        with torch.no_grad(): # Disable gradient calculations
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                asana_labels = labels // RATING_CLASSES
                rating_labels = labels % RATING_CLASSES

                asana_out, rating_out = model(images)
                loss_asana = criterion_asana(asana_out, asana_labels)
                loss_rating = criterion_rating(rating_out, rating_labels)
                loss = loss_asana + loss_rating

                val_loss += loss.item() * images.size(0)
                _, predicted_asana = torch.max(asana_out.data, 1)
                _, predicted_rating = torch.max(rating_out.data, 1)
                total_val += labels.size(0)
                correct_asana_val += (predicted_asana == asana_labels).sum().item()
                correct_rating_val += (predicted_rating == rating_labels).sum().item()

        # --- Calculate Validation Statistics ---
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_asana_acc = 100 * correct_asana_val / total_val
        epoch_val_rating_acc = 100 * correct_rating_val / total_val
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Asana Acc: {epoch_val_asana_acc:.2f}% | "
              f"Val Rating Acc: {epoch_val_rating_acc:.2f}%")
    else:
        print(f"Epoch {epoch+1}/{EPOCHS} | No validation data to evaluate.")


# --- Save the Final Model ---
checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME)
try:
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\nTraining finished.")
    print(f"Model saved successfully to {checkpoint_path}")
except Exception as e:
    print(f"\nTraining finished.")
    print(f"Error saving model: {e}")
