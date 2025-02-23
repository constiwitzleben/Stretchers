import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import numpy as np
import os
from util.Affine_Transformations import apply_corotated_strain_with_keypoints, generate_strain_tensors, generate_27_strain_tensors
from models import Embedded_Conditional_Residual_MLP, Embedded_Conditional_Fully_Residual_MLP
import time
from matchers.max_similarity import StretcherDualSoftMaxMatcher
from util.matching import draw_matches, draw_matching_comparison, draw_matches_with_scores
from util.dedode import detect_and_describe, get_affine_deformed_descriptions
from util.image import draw_keypoints
from fenics_2D_elasticity_grid_proj import create_deformed_medical_image_pair, track_pixel_displacement


device = device = get_best_device()
detector = dedode_detector_L(weights = torch.load("models/dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("models/dedode_descriptor_B.pth", map_location = device))
matcher = DualSoftMaxMatcher()
stretcher_matcher = StretcherDualSoftMaxMatcher()

num_keypoints = 8
only27 = False
good_match_threshold = 3

im_path = 'data/medical_deformed/brain.png'
deformed_im_path = 'data/medical_deformed/brain_deformed.png'

u, new_lx, new_ly = create_deformed_medical_image_pair(im_path, deformed_im_path)

image = Image.open(im_path)
deformed_image = Image.open(deformed_im_path)
W, H = image.size
dW, dH = deformed_image.size
image = np.array(image, dtype=np.uint8)
if image.shape[-1] == 4:
    image = image[:,:,:3]
deformed_image = np.array(deformed_image, dtype=np.uint8)
if deformed_image.shape[-1] == 4:
    deformed_image = deformed_image[:,:,:3]

base_keypoints, base_P, base_descriptions = detect_and_describe(image, detector, descriptor, device, num_keypoints)
pixel_base_keypoints = detector.to_pixel_coords(base_keypoints.cpu(), H, W)
cv2.imwrite("Visualisations/base_keypoints.png", draw_keypoints(image, pixel_base_keypoints[0]))

pixel_deformed_keypoints = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in pixel_base_keypoints[0].cpu()])[None]
deformed_keypoints = detector.to_normalized_coords(torch.tensor(pixel_deformed_keypoints), dH, dW).to(torch.float32)

# Extract Deformed Keypoint Descriptions
im = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(deformed_image).resize((dW,dH)))/255.).permute(2,0,1)).float().to(device)[None]
batch = {"image": im}
deformed_descriptions = descriptor.describe_keypoints(batch, deformed_keypoints.to(device))['descriptions'].squeeze()
cv2.imwrite("Visualisations/deformed_keypoints.png", draw_keypoints(deformed_image, pixel_deformed_keypoints[0]))

normed1 = torch.nn.functional.normalize(base_descriptions, p=2, dim=1)
normed2 = torch.nn.functional.normalize(deformed_descriptions, p=2, dim=1)
print(torch.mm(normed1, normed2.T))