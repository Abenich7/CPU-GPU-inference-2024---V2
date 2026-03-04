import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

# ----------------------- Config -----------------------
dataset_dir = Path("/home/benilla/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images/Images")
output_dir = Path("/home/workspace/benilla/project/stanford_dogs_calib_subset_flat")
total_sample = 5000
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
# ------------------------------------------------------

def get_images_by_class(dataset_path: Path, exts=image_extensions) -> dict:
    """Collect images by class from dataset directory."""
    images_by_class = defaultdict(list)
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            images = [p for p in class_dir.iterdir() if p.suffix.lower() in exts]
            if images:
                images_by_class[class_dir.name] = images
    return images_by_class

def sample_images(images_by_class: dict, total_sample: int) -> list:
    """Sample images proportionally to class size and flatten into a single list."""
    total_images = sum(len(imgs) for imgs in images_by_class.values())
    sampled_images = []
    for cls, imgs in images_by_class.items():
        proportion = len(imgs) / total_images
        n_samples = min(len(imgs), round(proportion * total_sample))
        sampled_images.extend(random.sample(imgs, n_samples))
    return sampled_images

def save_flattened_images(sampled_images: list, output_path: Path):
    """Save all sampled images into a single folder with unique filenames."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    for idx, img_path in enumerate(sampled_images):
        # Prepend index to ensure unique filenames
        new_name = f"{idx}_{img_path.name}"
        shutil.copy2(img_path, output_path / new_name)

# ----------------------- Main -----------------------
images_by_class = get_images_by_class(dataset_dir)
print(f"Found {len(images_by_class)} classes.")

sampled_images = sample_images(images_by_class, total_sample)
print(f"Total sampled images: {len(sampled_images)}")

save_flattened_images(sampled_images, output_dir)
print(f"Saved sampled images to {output_dir}")
