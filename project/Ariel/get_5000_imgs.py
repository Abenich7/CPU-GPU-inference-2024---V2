import os
import shutil
import random
from collections import defaultdict

dataset_dir = "/home/benilla/.cache/kagglehub/datasets/jessicali9530/stanford-dogs-dataset/versions/2/images/Images"
output_dir = "/home/workspace/benilla/project/stanford_dogs_calib_subset"


images_by_class = defaultdict(list)
for root, dirs, files in os.walk(dataset_dir):
    for d in dirs:
        class_dir = os.path.join(root, d)
        class_images = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))
        ]
        images_by_class[d] = class_images

print(f"Found {len(images_by_class)} classes")

total_sample = 5000
class_counts = {cls: len(imgs) for cls, imgs in images_by_class.items()}
total_images = sum(class_counts.values())

sampled_images_by_class = defaultdict(list)
for cls, imgs in images_by_class.items():
    proportion = len(imgs) / total_images
    n_samples = int(round(proportion * total_sample))
    n_samples = min(n_samples, len(imgs))
    sampled_images_by_class[cls] = random.sample(imgs, n_samples)

# Flatten to list if needed
sampled_images = [img for cls_imgs in sampled_images_by_class.values() for img in cls_imgs]
print(f"Total sampled images: {len(sampled_images)}")

for cls, imgs in sampled_images_by_class.items():
    class_out_dir = os.path.join(output_dir, cls)
    os.makedirs(class_out_dir, exist_ok=True)
    
    for src_path in imgs:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(class_out_dir, filename)
        shutil.copy2(src_path, dst_path)  # preserves metadata

print(f"Saved sampled images to {output_dir}")
