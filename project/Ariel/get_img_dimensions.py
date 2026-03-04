from PIL import Image
import os

def get_image_dimensions(filepath):
    try:
        with Image.open(filepath) as img:
            return img.size  # (width, height)
    except IOError:
        return None

dataset_dir = "/home/workspace/benilla/project/stanford_dogs_calib_subset"


min_area = float("inf")
max_area = 0

min_image = None
max_image = None

min_width = float("inf")
max_width = 0
min_height = float("inf")
max_height = 0

for root, _, files in os.walk(dataset_dir):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(root, filename)
            dims = get_image_dimensions(path)

            if dims:
                w, h = dims
                area = w * h

                # area-based min/max
                if area < min_area:
                    min_area = area
                    min_image = (path, dims)

                if area > max_area:
                    max_area = area
                    max_image = (path, dims)

                # per-dimension min/max
                min_width = min(min_width, w)
                max_width = max(max_width, w)
                min_height = min(min_height, h)
                max_height = max(max_height, h)

print("Smallest image (by area):", min_image)
print("Largest image (by area):", max_image)
print("Min width:", min_width)
print("Max width:", max_width)
print("Min height:", min_height)
print("Max height:", max_height)
import configparser

config = configparser.ConfigParser()

config["SMALLEST_IMAGE"] = {
    "path": min_image[0],
    "width": min_image[1][0],
    "height": min_image[1][1],
    "area": min_area
}

config["LARGEST_IMAGE"] = {
    "path": max_image[0],
    "width": max_image[1][0],
    "height": max_image[1][1],
    "area": max_area
}

config["DIMENSIONS"] = {
    "min_width": min_width,
    "max_width": max_width,
    "min_height": min_height,
    "max_height": max_height
}

with open("stanford_dogs.cfg", "w") as f:
    config.write(f)
