# yoga_model/dataset.py
import os
import sys
from torch.utils.data import Dataset
from PIL import Image
# Removed torchvision transforms import, as it's applied in train/test scripts

class YogaDataset(Dataset):
    def _init_(self, root_dir, transform=None):
        """
        Initializes the dataset.
        Assumes directory structure: root_dir/{asana_name}/{rating_name}/{frame_folder}/{frame_file}.jpg
        Generates a single integer label: asana_idx * num_rating_classes + rating_idx
        """
        self.samples = []
        self.transform = transform
        self.root_dir = root_dir
        self.class_to_idx = {} # Maps combined 'asana_rating' string to the integer label
        self.asana_names = []
        self.rating_names_map = {} # Stores ratings found for each asana

        print(f"Scanning dataset in: {os.path.abspath(root_dir)}")

        if not os.path.isdir(root_dir):
             raise FileNotFoundError(f"Root directory not found: {root_dir}")

        # --- Discover Asanas and Ratings ---
        temp_asana_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not temp_asana_names:
            print(f"Warning: No subdirectories found in {root_dir}. Expected asana folders.")
            return # No data to load

        # Determine rating classes dynamically and consistently
        all_rating_names = set()
        for asana_name in temp_asana_names:
            asana_path = os.path.join(root_dir, asana_name)
            current_ratings = sorted([r for r in os.listdir(asana_path) if os.path.isdir(os.path.join(asana_path, r))])
            if current_ratings:
                self.asana_names.append(asana_name) # Only include asanas that have rating subfolders
                self.rating_names_map[asana_name] = current_ratings
                all_rating_names.update(current_ratings)
            else:
                print(f"Warning: No rating subfolders found in {asana_path}. Skipping this asana.")

        if not self.asana_names:
            print(f"Error: No valid asana folders with rating subfolders found in {root_dir}.")
            return

        # Establish a canonical order for ratings across all asanas
        self.canonical_rating_names = sorted(list(all_rating_names))
        self.num_rating_classes = len(self.canonical_rating_names)

        print(f"Found Asana folders: {self.asana_names}")
        print(f"Canonical Rating folders: {self.canonical_rating_names} (Number: {self.num_rating_classes})")

        # --- Build Samples and Class Mapping ---
        for asana_idx, asana_name in enumerate(self.asana_names):
            asana_path = os.path.join(root_dir, asana_name)
            # Use the canonical rating names for consistent indexing
            for rating_idx, rating_name in enumerate(self.canonical_rating_names):
                rating_path = os.path.join(asana_path, rating_name)

                # Check if this specific rating exists for this asana
                if not os.path.isdir(rating_path):
                    # print(f"Debug: Rating '{rating_name}' not found in '{asana_name}', skipping.")
                    continue # Skip if this specific asana doesn't have this rating type

                # Calculate the combined integer label
                label = asana_idx * self.num_rating_classes + rating_idx
                combined_class_name = f"{asana_name}_{rating_name}"
                self.class_to_idx[combined_class_name] = label

                # Iterate through frame folders and actual frames
                try:
                    for item_in_rating_dir in os.listdir(rating_path):
                        item_path = os.path.join(rating_path, item_in_rating_dir)
                        # Check if it's the video frame folder
                        if os.path.isdir(item_path):
                            frame_folder_path = item_path
                            for frame_filename in os.listdir(frame_folder_path):
                                # Check if it's likely an image file
                                if frame_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                    img_path = os.path.join(frame_folder_path, frame_filename)
                                    self.samples.append((img_path, label))
                        # Optional: Handle if frames are directly under rating folder (unlikely based on structure)
                        # elif item_in_rating_dir.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        #     img_path = item_path
                        #     self.samples.append((img_path, label))

                except FileNotFoundError:
                     print(f"Warning: Folder not found or inaccessible during scan: {rating_path}")
                except Exception as e:
                     print(f"Warning: Error scanning folder {rating_path}: {e}")

        # Store class names based on the generated integer labels for consistency check
        # Sort by index to match the order labels will likely appear
        self.class_names = [name for name, idx in sorted(self.class_to_idx.items(), key=lambda item: item[1])]
        print(f"Finished scanning. Found {len(self.samples)} samples.")
        print(f"Final classes mapped to indices: {self.class_to_idx}")
        print(f"Total unique combined classes found: {len(self.class_names)}")


    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        if idx >= len(self.samples):
             raise IndexError("Index out of range")
        img_path, label = self.samples[idx]
        try:
            # Ensure image is closed properly using 'with'
            with Image.open(img_path) as image:
                 # Ensure conversion to RGB happens after opening
                 image = image.convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, label
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_path} (referenced by index {idx}). Check dataset integrity.", file=sys.stderr)
            raise FileNotFoundError(f"Image file not found: {img_path}")
        except Exception as e:
            print(f"Error loading or transforming image {img_path}: {e}", file=sys.stderr)
            raise e # Re-raise the exception
