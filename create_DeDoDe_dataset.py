import numpy
import os
import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import cv2
import numpy as np
from Affine_Transformations import generate_strain_tensors, apply_corotated_strain_with_keypoints
import time

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Define parameters
image_dir = "data/data"
deformed_dir = "data/transformed_data"
deformations_per_image = 8
kp_per_deformation = 8
kp_per_image = deformations_per_image * kp_per_deformation
num_images = 1000
margin = 0.2
samples = 4000

# Create detector and descriptor instances
device = get_best_device()
detector = dedode_detector_L(weights = torch.load("dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("dedode_descriptor_B.pth", map_location = device))

# Load image_paths
image_names = os.listdir(image_dir)
# image_paths = sorted([os.path.join(image_dir, file) for file in image_names])

# Initialize arrays to store descriptors
non_deformed_descriptors = np.zeros((len(image_names)*deformations_per_image*kp_per_deformation, 256))
deformed_descriptors = np.zeros((len(image_names)*deformations_per_image*kp_per_deformation, 256))
deformation_idx = np.random.randint(0,64,size=(num_images,deformations_per_image))
deformation_grid = generate_strain_tensors()

# Loop over image_paths
for i, image_name in enumerate(image_names):

    # Extract image info
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)
    W, H = image.size
    np_image = np.array(image)

    # Extract inner image
    np_inner_image = np_image[H//4:3*H//4, W//4:3*W//4, :]
    h, w = np_inner_image.shape[:2]

    # Create Padded image
    hpadding = 3*H
    wpadding = 3*W
    padded_np_image = np.pad(np_image, ((hpadding, hpadding), (wpadding, wpadding), (0, 0)), mode='reflect')

    # Detect keypoints
    np_inner_image = np.array(np_inner_image,dtype=np.uint8)
    non_deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(np_inner_image).resize((w,h)))/255.).permute(2,0,1)).float().to(device)[None]
    batch = {"image": non_deformed_image}
    detections = detector.detect(batch, num_keypoints = samples)
    keypoints, P = detections["keypoints"], detections["confidence"]
    
    # Chose the best keypoints within the margin or generate random ones
    keypoints = keypoints[0]
    condition = (np.abs(keypoints.cpu()[:, 0]) <= margin) & (np.abs(keypoints.cpu()[:, 1]) <= margin)
    keypoints = keypoints[condition]
    if len(keypoints) > kp_per_image:
        keypoints = keypoints[:kp_per_image]
    else:
        keypoints = (torch.rand((kp_per_image, 2)) * 2 - 1) * margin
    indices = torch.randperm(keypoints.size(0))
    keypoints = keypoints[indices]

    # Store non-deformed descriptors
    description = descriptor.describe_keypoints(batch, keypoints[None,...].to(device))["descriptions"].squeeze()
    non_deformed_descriptors[i*kp_per_deformation*deformations_per_image:(i+1)*kp_per_deformation*deformations_per_image] = description.cpu()

    # Extract Deformations
    deformations = np.array(deformation_grid)[deformation_idx[i,:]]

    # Reshape keypoints
    keypoints = keypoints.reshape(deformations_per_image, kp_per_deformation, 2)

    # Loop over deformations
    for j, (deformation, keypoint_set) in enumerate(zip(deformations,keypoints)):

        # Convert keypoint to pixel coordinates
        pixel_keypoint_set = detector.to_pixel_coords(keypoint_set.cpu(), h, w)
        whole_pixel_keypoint_set = pixel_keypoint_set + torch.tensor([W//4, H//4]) + torch.tensor([wpadding, hpadding])
        
        # Apply deformation
        deformed_padded_image, whole_pixel_deformed_keypoint_set = apply_corotated_strain_with_keypoints(padded_np_image, whole_pixel_keypoint_set, deformation)
        pixel_deformed_keypoint_set = whole_pixel_deformed_keypoint_set - np.array([W//4, H//4]) - np.array([wpadding, hpadding])
        deformed_keypoint_set = detector.to_normalized_coords(torch.tensor(pixel_deformed_keypoint_set), h, w).to(torch.float32)[0]

        # Get deformed descriptor without saving and reading the image
        deformed_padded_image = np.array(deformed_padded_image, dtype=np.uint8)
        deformed_image = deformed_padded_image[hpadding:-hpadding, wpadding:-wpadding, :]
        inner_deformed_image = deformed_image[H//4:3*H//4, W//4:3*W//4, :]
        inner_deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(inner_deformed_image).resize((w,h)))/255.).permute(2,0,1)).float().to(device)[None]
        deformed_batch = {"image": inner_deformed_image}
        deformed_description_set = descriptor.describe_keypoints(deformed_batch, deformed_keypoint_set[None,...].to(device))["descriptions"].squeeze()

        # deformed_descriptors[i*kp_per_deformation*deformations_per_image + j*deformations_per_image + k] = deformed_description_set.cpu()
        deformed_descriptors[i*kp_per_deformation*deformations_per_image + j*kp_per_deformation:i*kp_per_deformation*deformations_per_image + (j+1)*kp_per_deformation] = deformed_description_set.cpu()

    if i % 10 == 0:
        print(f"Processed {i} images")
            
        
# Save descriptors
torch_descriptors = torch.tensor(non_deformed_descriptors)
torch_deformed_descriptors = torch.tensor(deformed_descriptors)
torch_deformations = torch.tensor(deformation_idx)
torch.save({'descriptors': torch_descriptors, 'deformed_descriptors': torch_deformed_descriptors, 'transformations': torch_deformations}, "data/DeDoDe_Descriptors_Dataset.pth")