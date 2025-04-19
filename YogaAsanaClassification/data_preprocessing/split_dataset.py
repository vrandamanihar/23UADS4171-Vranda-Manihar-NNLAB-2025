import os
import shutil
import random
from pathlib import Path

INPUT_DIR = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\resized_frames"
OUTPUT_DIR = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\final_dataset"


SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

def split_dataset():
    for asana in os.listdir(INPUT_DIR):
        for quality in os.listdir(os.path.join(INPUT_DIR, asana)):
            data_path = os.path.join(INPUT_DIR, asana, quality)
            video_folders = os.listdir(data_path)
            random.shuffle(video_folders)

            total = len(video_folders)
            train_cut = int(total * SPLIT_RATIOS[0])
            val_cut = train_cut + int(total * SPLIT_RATIOS[1])

            splits = {
                'train': video_folders[:train_cut],
                'val': video_folders[train_cut:val_cut],
                'test': video_folders[val_cut:]
            }

            for split in splits:
                for folder in splits[split]:
                    src = os.path.join(data_path, folder)
                    dst = os.path.join(OUTPUT_DIR, split, asana, quality, folder)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copytree(src, dst)
                    print(f"Copied {src} to {dst}")

if __name__ == "__main__":
    split_dataset()
