import cv2
import os
from pathlib import Path

INPUT_DIR = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\raw_frames"
OUTPUT_DIR = r"C:\Users\vrand\23UADS4171-Vranda-Manihar-NNLAB-2025-3\YogaAsanaClassification\processed_data\resized_frames"

IMG_SIZE = (224, 224)

def process_and_save_image(image_path, output_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_bgr = (img_normalized * 255).astype('uint8')
    cv2.imwrite(output_path, img_bgr)

def resize_all():
    for asana in os.listdir(INPUT_DIR):
        for quality in os.listdir(os.path.join(INPUT_DIR, asana)):
            for video_folder in os.listdir(os.path.join(INPUT_DIR, asana, quality)):
                input_video_folder = os.path.join(INPUT_DIR, asana, quality, video_folder)
                output_video_folder = os.path.join(OUTPUT_DIR, asana, quality, video_folder)
                os.makedirs(output_video_folder, exist_ok=True)

                for frame in os.listdir(input_video_folder):
                    in_frame_path = os.path.join(input_video_folder, frame)
                    out_frame_path = os.path.join(output_video_folder, frame)
                    process_and_save_image(in_frame_path, out_frame_path)

if __name__ == "__main__":
    resize_all()
