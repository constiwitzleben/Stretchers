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

# Define parameters
image_dir = "data/data"
deformed_dir = "data/transformed_data"
deformations_per_kp = 8
kp_per_image = 8
num_images = 1000
margin = 0.5
samples = 100

# Create detector and descriptor instances
device = get_best_device()
detector = dedode_detector_L(weights = torch.load("dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("dedode_descriptor_B.pth", map_location = device))

# Load image_paths
image_names = os.listdir(image_dir)
# image_paths = sorted([os.path.join(image_dir, file) for file in image_names])

# Initialize arrays to store descriptors
non_deformed_descriptors = np.zeros((len(image_names)*kp_per_image*deformations_per_kp, 256))
deformed_descriptors = np.zeros((len(image_names)*kp_per_image*deformations_per_kp, 256))
deformation_idx = np.random.randint(0,64,size=(num_images,kp_per_image,deformations_per_kp))
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

    # Convert keypoints to pixel coordinates
    pixel_keypoints = detector.to_pixel_coords(keypoints.cpu(), H, W)

    # Store non-deformed descriptors
    print(keypoints)
    description = descriptor.describe_keypoints_from_path(image_path, keypoints[None,...])["descriptions"].squeeze()
    non_deformed_descriptors[i*kp_per_image*deformations_per_kp:(i+1)*kp_per_image*deformations_per_kp] = np.repeat(description.cpu(),deformations_per_kp,axis=0)

    # Loop over keypoints
    for j, kp in enumerate(keypoints.cpu()):
        
        # Get deformations for this keypoint
        deformations = np.array(deformation_grid)[deformation_idx[i,j,:]]

        # Loop over deformations
        for k, deformation in enumerate(deformations):

            # Prep Image
            np_image = np.array(image)[...,::-1]
            
            # Apply deformation
            deformed_image, pixel_deformed_keypoints = apply_corotated_strain_with_keypoints(np_image, pixel_keypoints, deformation)
            deformed_keypoints = detector.to_normalized_coords(torch.tensor(pixel_deformed_keypoints), H, W).to(torch.float32)[0]
            deformed_image_path = os.path.join(deformed_dir, image_name)
            cv2.imwrite(deformed_image_path, deformed_image)

            # Store deformed descriptors
            print(deformed_keypoints)
            deformed_description = descriptor.describe_keypoints_from_path(deformed_image_path, deformed_keypoints[None,...].to(device))["descriptions"].squeeze()
            deformed_descriptors[i*kp_per_image*deformations_per_kp + j*deformations_per_kp + k] = deformed_description.cpu()
            break
        break
    break