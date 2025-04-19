import torch
import torch.nn as nn
import torch.nn.functional as F

class YogaCNN(nn.Module):
    def __init__(self, num_classes=6, quality_classes=3):
        super(YogaCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Automatically calculate the flatten size
        self._to_linear = None
        self._get_flattened_size()

        # Fully connected layers
        self.fc = nn.Linear(self._to_linear, 256)
        self.asana_classifier = nn.Linear(256, num_classes)
        self.quality_classifier = nn.Linear(256, quality_classes)

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc(x))

        asana_output = self.asana_classifier(x)
        quality_output = self.quality_classifier(x)

        return asana_output, quality_output
