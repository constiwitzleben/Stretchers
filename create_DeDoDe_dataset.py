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
samples = 1000

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

    # Detect keypoints
    detections = detector.detect_from_path(image_path, num_keypoints = samples)
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
    # print(keypoints)
    description = descriptor.describe_keypoints_from_path(image_path, keypoints[None,...].to(device))["descriptions"].squeeze()
    non_deformed_descriptors[i*kp_per_deformation*deformations_per_image:(i+1)*kp_per_deformation*deformations_per_image] = description.cpu()

    # Extract Deformations
    deformations = np.array(deformation_grid)[deformation_idx[i,:]]

    keypoints = keypoints.reshape(deformations_per_image, kp_per_deformation, 2)

    # Loop over deformations
    for j, (deformation, keypoint_set) in enumerate(zip(deformations,keypoints)):

        # Convert keypoint to pixel coordinates
        pixel_keypoint_set = detector.to_pixel_coords(keypoint_set.cpu(), H, W)
        
        # Prep Image
        np_image = np.array(image)[...,::-1]
        
        # Apply deformation
        deformed_image, pixel_deformed_keypoint_set = apply_corotated_strain_with_keypoints(np_image, pixel_keypoint_set, deformation)
        deformed_keypoint_set = detector.to_normalized_coords(torch.tensor(pixel_deformed_keypoint_set), H, W).to(torch.float32)[0]
        # deformed_image_path = os.path.join(deformed_dir, image_name)
        # cv2.imwrite(deformed_image_path, deformed_image)

        # Get deformed descriptor without saving and reading the image
        deformed_image = np.array(deformed_image, dtype=np.uint8)
        deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(deformed_image).resize((W,H)))/255.).permute(2,0,1)).float().to(device)[None]
        batch = {"image": deformed_image}
        deformed_description_set = descriptor.describe_keypoints(batch, deformed_keypoint_set[None,...].to(device))["descriptions"].squeeze()

        # Store deformed descriptors
        # print(deformed_keypoint)
        # deformed_description = descriptor.describe_keypoints_from_path(deformed_image_path, deformed_keypoint[None,...].to(device))["descriptions"].squeeze()

        # deformed_descriptors[i*kp_per_deformation*deformations_per_image + j*deformations_per_image + k] = deformed_description_set.cpu()
        deformed_descriptors[i*kp_per_deformation*deformations_per_image + j*kp_per_deformation:i*kp_per_deformation*deformations_per_image + (j+1)*kp_per_deformation] = deformed_description_set.cpu()

    if i % 10 == 0:
        print(f"Processed {i} images")
            
        
# Save descriptors
torch_descriptors = torch.tensor(non_deformed_descriptors)
torch_deformed_descriptors = torch.tensor(deformed_descriptors)
torch_deformations = torch.tensor(deformation_idx)
torch.save({'descriptors': torch_descriptors, 'deformed_descriptors': torch_deformed_descriptors, 'transformations': torch_deformations}, "data/DeDoDe_Descriptors_Dataset.pth")