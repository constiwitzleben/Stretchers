import numpy
import matplotlib.pyplot as plt
import os
import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import cv2
import numpy as np
from util.Affine_Transformations import generate_strain_tensors, apply_corotated_strain_with_keypoints
import time
from util.superpoint import sp_detect_and_describe, custom_sample_descriptors
from util.image import draw_keypoints

# def draw_kpts(im, kpts):    
#     kpts = [cv2.KeyPoint(float(x),float(y),15.) for x,y in kpts]
#     im = np.array(im)
#     ret = cv2.drawKeypoints(im, kpts, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#     return ret

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Define parameters
image_dir = "data/medical_training_data"
deformations_per_image = 14
kp_per_deformation = 14
kp_per_image = deformations_per_image * kp_per_deformation
num_images = 694

# Create detector and descriptor instances
device = get_best_device()

# Load image_paths
image_names = os.listdir(image_dir)

# Initialize arrays to store descriptors
non_deformed_descriptors = np.zeros((len(image_names)*deformations_per_image*kp_per_deformation, 256))
deformed_descriptors = np.zeros((len(image_names)*deformations_per_image*kp_per_deformation, 256))
deformation_grid = generate_strain_tensors()
deformation_idx = np.random.randint(0,len(deformation_grid),size=(num_images,deformations_per_image))

# Loop over image_paths
for i, image_name in enumerate(image_names):

    # Extract image info
    image_path = os.path.join(image_dir, image_name)
    image = Image.open(image_path)
    W, H = image.size
    image = image.resize((W // 2, H // 2))
    image = np.array(image, dtype=np.uint8)

    # Extract Keypoints and Descriptors
    keypoints, scores, descriptions, dense_descriptions, scales, img_size = sp_detect_and_describe(image, device, kp_per_image)
    top_indices = scores.argsort(descending=True)[:kp_per_image]
    keypoints = keypoints[top_indices]
    descriptions = descriptions[top_indices]

    # Store non-deformed descriptors
    non_deformed_descriptors[i*kp_per_deformation*deformations_per_image:(i+1)*kp_per_deformation*deformations_per_image] = descriptions.cpu()

    # Extract Deformations
    deformations = np.array(deformation_grid)[deformation_idx[i,:]]

    # original_image = draw_kpts(image, keypoints)
    # Reshape keypoints
    keypoints = keypoints.reshape(deformations_per_image, kp_per_deformation, 2)

    # Loop over deformations
    for j, (deformation, keypoint_set) in enumerate(zip(deformations,keypoints)):

        # Apply deformation
        deformed_image, deformed_keypoint_set = apply_corotated_strain_with_keypoints(image, keypoint_set, deformation, dataset_mode=False)
        
        # deformed_image = draw_kpts(deformed_image, deformed_keypoint_set[0])
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        # ax[0].imshow(original_image)
        # ax[1].imshow(deformed_image)
        # plt.show()

        # Get deformed descriptor without saving and reading the image
        _, _, _, deformed_dense_descriptions, scales, _ = sp_detect_and_describe(deformed_image, device, 10000)
        sp_keypoints = (torch.tensor(deformed_keypoint_set[0]) + 0.5) * scales - 0.5
        deformed_description_set = custom_sample_descriptors(sp_keypoints.to(torch.float32), deformed_dense_descriptions.cpu()).permute(0,2,1)[0]

        # deformed_descriptors[i*kp_per_deformation*deformations_per_image + j*deformations_per_image + k] = deformed_description_set.cpu()
        deformed_descriptors[i*kp_per_deformation*deformations_per_image + j*kp_per_deformation:i*kp_per_deformation*deformations_per_image + (j+1)*kp_per_deformation] = deformed_description_set.cpu()

    if i % 10 == 0:
        print(f"Processed {i} images")
            
        
# Save descriptors
torch_descriptors = torch.tensor(non_deformed_descriptors)
torch_deformed_descriptors = torch.tensor(deformed_descriptors)
torch_deformations = torch.tensor(deformation_idx)
torch.save({'descriptors': torch_descriptors, 'deformed_descriptors': torch_deformed_descriptors, 'transformations': torch_deformations}, "data/SuperPoint_Descriptors_Dataset.pth")