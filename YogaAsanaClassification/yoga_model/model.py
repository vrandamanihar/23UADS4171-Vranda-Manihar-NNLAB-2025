# yoga_model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class YogaCNN(nn.Module):
    def _init_(self, asana_classes=6, rating_classes=3):
        super(YogaCNN, self)._init_()

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), # Input: (N, 3, 224, 224) -> Output: (N, 32, 224, 224)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),     # Output: (N, 32, 112, 112)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),# Output: (N, 64, 112, 112)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),     # Output: (N, 64, 56, 56)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# Output: (N, 128, 56, 56)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)      # Output: (N, 128, 28, 28)
        )

        # Calculate the flattened size dynamically or use the known value
        # Known value: 128 channels * 28 height * 28 width = 100352
        self.flattened_size = 128 * 28 * 28

        self.flatten = nn.Flatten() # Flattens the output of features to (N, 100352)

        # --- Corrected Linear Layers ---
        # Input features must match the flattened_size
        self.fc_asana = nn.Linear(self.flattened_size, asana_classes)
        self.fc_rating = nn.Linear(self.flattened_size, rating_classes)

        # --- Optional: Add more layers or dropout ---
        # Example: Add another hidden layer and dropout
        # self.fc1 = nn.Linear(self.flattened_size, 512)
        # self.dropout = nn.Dropout(0.5)
        # self.fc_asana = nn.Linear(512, asana_classes)
        # self.fc_rating = nn.Linear(512, rating_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x) # Shape: (N, 100352)

        # --- Pass through corrected linear layers ---
        asana_output = self.fc_asana(x)
        rating_output = self.fc_rating(x)

        # --- If using optional extra layers: ---
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # asana_output = self.fc_asana(x)
        # rating_output = self.fc_rating(x)

        return asana_output,rating_output
