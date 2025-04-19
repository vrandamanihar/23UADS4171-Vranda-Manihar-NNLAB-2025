import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

class YogaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        for class_idx, asana in enumerate(self.classes):
            asana_dir = os.path.join(root_dir, asana)
            for img_name in os.listdir(asana_dir):
                self.samples.append((os.path.join(asana_dir, img_name), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label