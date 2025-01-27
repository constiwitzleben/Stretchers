import numpy
import os
import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import cv2
import numpy as np
from Affine_Transformations import generate_strain_tensors

image_dir = "data/data"
deformations_per_kp = 8
kp_per_image = 8
num_images = 1000

device = get_best_device()
detector = dedode_detector_L(weights = torch.load("dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("dedode_descriptor_B.pth", map_location = device))

images = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir)])

non_deformed_descriptors = np.zeros((len(images)*kp_per_image*deformations_per_kp, 256))
deformed_descriptors = np.zeros((len(images)*kp_per_image*deformations_per_kp, 256))
deformation_idx = np.random.randint(0,64,size=(num_images,kp_per_image,deformations_per_kp))
deformation_grid = generate_strain_tensors()

for i, image in enumerate(images):

    detections = detector.detect_from_path(image, num_keypoints = kp_per_image)
    keypoints, P = detections["keypoints"], detections["confidence"]
    description = descriptor.describe_keypoints_from_path(image, keypoints)["descriptions"].squeeze()
    non_deformed_descriptors[i*kp_per_image*deformations_per_kp:(i+1)*kp_per_image*deformations_per_kp] = np.repeat(description.cpu(),deformations_per_kp,axis=0)

    for j, kp in enumerate(keypoints):
        
        deformations = deformation_grid[deformation_idx[i,j]]

        for deformation in enumerate(deformations):
            
            

            break
        break
    # break