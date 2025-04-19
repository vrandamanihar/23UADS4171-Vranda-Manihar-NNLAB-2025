import cv2
import os
from pathlib import Path

# Settings
INPUT_DIR = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\dataset"
OUTPUT_DIR = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\raw_frames"

FRAMES_PER_VIDEO = 15

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frame_count // FRAMES_PER_VIDEO)
    
    count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0 and count < FRAMES_PER_VIDEO:
            frame_name = f"{count:03d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(frame_path, frame)
            count += 1
        frame_idx += 1

    cap.release()

def extract_all():
    for asana in os.listdir(INPUT_DIR):
        for quality in os.listdir(os.path.join(INPUT_DIR, asana)):
            folder_path = os.path.join(INPUT_DIR, asana, quality)
            for video_file in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_file)
                output_folder = os.path.join(OUTPUT_DIR, asana, quality, Path(video_file).stem)
                os.makedirs(output_folder, exist_ok=True)
                extract_frames(video_path, output_folder)
                print(f"Extracted frames from: {video_path}")

if __name__ == "__main__":
    extract_all()
