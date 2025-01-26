import os
import random
from PIL import Image
import numpy as np

def sample_image_patches(image_dir, output_dir, patch_size=(16, 16), patches_per_image=64):
    """
    Samples random patches from images in a directory.

    Args:
        image_dir (str): Directory containing the images.
        patch_size (tuple): Dimensions (width, height) of the patches.
        patches_per_image (int): Number of patches to sample per image.

    Returns:
        list: A list of sampled patches as numpy arrays.
    """
    # Get a list of image file paths
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    sampled_patches = []

    for image_file in image_files:
        # Open the image
        with Image.open(image_file) as img:
            img = img.convert('RGB')  # Ensure 3-channel images
            width, height = img.size

            for _ in range(patches_per_image):
                # Randomly sample the top-left corner of the patch
                if width > patch_size[0] and height > patch_size[1]:
                    x = random.randint(0, width - patch_size[0])
                    y = random.randint(0, height - patch_size[1])
                    patch = img.crop((x, y, x + patch_size[0], y + patch_size[1]))
                    sampled_patches.append(np.array(patch))

    for i, patch in enumerate(sampled_patches):
        patch_image = Image.fromarray(patch)
        patch_image.save(os.path.join(output_dir, f"patch_{i}.png"))

    print(f"Sampled {len(sampled_patches)} patches and saved them to {output_directory}.")

    return sampled_patches

# Parameters
image_directory = "data/data"  # Replace with the directory containing your 100 images
output_directory = "data/patches"  # Replace with where you want to save patches (optional)
# os.makedirs(output_directory, exist_ok=True)

# Sample patches
patches = sample_image_patches(image_directory, output_directory, patch_size=(32, 32), patches_per_image=64)