import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import numpy as np
import os
from util.Affine_Transformations import apply_corotated_strain_with_keypoints, generate_strain_tensors, generate_27_strain_tensors, generate_larger_strain_tensors
from models import Embedded_Conditional_Residual_MLP, Embedded_Conditional_Fully_Residual_MLP
import time
from matchers.max_similarity import StretcherDualSoftMaxMatcher
from util.matching import draw_matches, draw_matching_comparison, draw_matches_with_scores
from util.dedode import detect_and_describe, get_affine_deformed_descriptions
from util.image import draw_keypoints, visualize_keypoint_similarities
from util.superpoint import sp_detect_and_describe, custom_sample_descriptors, sp_get_affine_deformed_descriptions
from fenics_2D_elasticity_grid_proj import create_deformed_medical_image_pair, track_pixel_displacement
import math


device = device = get_best_device()
detector = dedode_detector_L(weights = torch.load("models/dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("models/dedode_descriptor_B.pth", map_location = device))
matcher = DualSoftMaxMatcher()
stretcher_matcher = StretcherDualSoftMaxMatcher()

num_keypoints = 10000
stretch_type = 'normal'
good_match_threshold = 5
chosen_keypoints = None
model = 'superpoint'

if stretch_type == 'only27':
    tensors = np.array(generate_27_strain_tensors())
elif stretch_type == 'larger':
    tensors = np.array(generate_larger_strain_tensors())
else:
    tensors = np.array(generate_strain_tensors())

im_path = 'data/medical_deformed/brain_lowres.png'
deformed_im_path = 'data/medical_deformed/brain_lowres_deformed.png'
# im_path = 'data/medical_deformed/skull.png'
# deformed_im_path = 'data/medical_deformed/skull_deformed.png'

u, new_lx, new_ly = create_deformed_medical_image_pair(im_path, deformed_im_path)

image = Image.open(im_path)
deformed_image = Image.open(deformed_im_path)
W, H = image.size
print(W,H)
dW, dH = deformed_image.size
print(dW,dH)
image = np.array(image, dtype=np.uint8)
if image.shape[-1] == 4:
    image = image[:,:,:3]
deformed_image = np.array(deformed_image, dtype=np.uint8)
if deformed_image.shape[-1] == 4:
    deformed_image = deformed_image[:,:,:3]

if model == 'dedode':
    base_keypoints, base_P, base_descriptions = detect_and_describe(image, detector, descriptor, device, num_keypoints)
    pixel_base_keypoints = detector.to_pixel_coords(base_keypoints.cpu(), H, W)[0]
elif model == 'superpoint':
    pixel_base_keypoints, base_P, base_descriptions, dense_descriptions, _ = sp_detect_and_describe(image, device, num_keypoints)

# print(stop)
cv2.imwrite("Visualisations/base_keypoints.png", draw_keypoints(image, pixel_base_keypoints))

pixel_deformed_keypoints = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in pixel_base_keypoints.cpu()])[None]
deformed_keypoints = detector.to_normalized_coords(torch.tensor(pixel_deformed_keypoints), dH, dW).to(torch.float32)

cv2.imwrite("Visualisations/deformed_keypoints.png", draw_keypoints(deformed_image, pixel_deformed_keypoints[0]))

# Extract Deformed Keypoint Descriptions
if  model == 'dedode':
    im = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(deformed_image).resize((dW,dH)))/255.).permute(2,0,1)).float().to(device)[None]
    batch = {"image": im}
    deformed_descriptions = descriptor.describe_keypoints(batch, deformed_keypoints.to(device))['descriptions'].squeeze()
elif model == 'superpoint':
    _, _, _, deformed_dense_descriptions, scales = sp_detect_and_describe(deformed_image, device, num_keypoints)
    sp_keypoints = (torch.tensor(pixel_deformed_keypoints[0]) + 0.5) * scales - 0.5
    deformed_descriptions = custom_sample_descriptors(sp_keypoints, deformed_dense_descriptions.cpu()).permute(0,2,1)[0]

normed1 = torch.nn.functional.normalize(base_descriptions, p=2, dim=1)
normed2 = torch.nn.functional.normalize(deformed_descriptions, p=2, dim=1)
similarity_matrix = torch.mm(normed1, normed2.T)
# print(similarity_matrix)
cosines = torch.diag(similarity_matrix)

deformed_image_vis, min_cos, max_cos = visualize_keypoint_similarities(deformed_image, pixel_deformed_keypoints[0], cosines)

# Save and display
cv2.imwrite("Visualisations/deformed_keypoints_similarity.png", deformed_image_vis)

if model == 'dedode':
    stretched_descriptions = torch.tensor(get_affine_deformed_descriptions(image, pixel_base_keypoints, tensors, detector, descriptor, device)).to(device)
elif model == 'superpoint':
    stretched_descriptions = sp_get_affine_deformed_descriptions(image, pixel_base_keypoints, tensors, device).to(torch.float32).to(device)
    print(stretched_descriptions.shape)

normed1 = torch.nn.functional.normalize(stretched_descriptions.cpu(), p=2, dim=2)
normed2 = torch.nn.functional.normalize(deformed_descriptions.cpu(), p=2, dim=1)
max_similarity_matrix = torch.full((len(deformed_descriptions), len(deformed_descriptions)), float('-inf'), device=normed1.device)
# Loop over batches
for i in range(normed1.shape[0]):  # Iterate over 125 batches
    sim_matrix = torch.matmul(normed1[i], normed2.T)  # Shape: (10000, 10000)
    max_similarity_matrix = torch.maximum(max_similarity_matrix, sim_matrix)  # Element-wise max update

# print(similarity_matrix)
cosines = torch.diag(max_similarity_matrix)

stretched_deformed_image_vis, _, _ = visualize_keypoint_similarities(deformed_image, pixel_deformed_keypoints[0], cosines, min_cos, max_cos)

# Save and display
cv2.imwrite("Visualisations/stretched_deformed_keypoints_similarity.png", stretched_deformed_image_vis)

if model == 'dedode':
    detected_deformed_keypoints, deformed_P, detected_deformed_descriptions = detect_and_describe(deformed_image, detector, descriptor, device, num_keypoints)
    pixel_detected_deformed_keypoints = detector.to_pixel_coords(detected_deformed_keypoints.cpu(), dH, dW)[0]
if model == 'superpoint':
    pixel_detected_deformed_keypoints, deformed_P, detected_deformed_descriptions, detected_deformed_dense_descriptions, scales = sp_detect_and_describe(deformed_image, device, num_keypoints)


stretched_matches, deformed_matches, batch_ids = stretcher_matcher.match(pixel_base_keypoints[None].to(device), stretched_descriptions.to(device),
        pixel_detected_deformed_keypoints[None].to(device), detected_deformed_descriptions.to(device),
        P_A = base_P, P_B = deformed_P,
        normalize = True, inv_temp=20, threshold = 0.01, stretch_type=stretch_type)

# if model == 'dedode':
#     stretched_matches, deformed_matches = matcher.to_pixel_coords(stretched_matches, deformed_matches, H, W, dH, dW)

print(stretched_matches.shape)
print(deformed_matches.shape)
# stretched_matches = stretched_matches.squeeze()
# deformed_mathces = deformed_matches.squeeze()

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in stretched_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'Accuracy after stretching: {accuracy} ({good} / {total})')

stretched_matches_image = Image.fromarray(draw_matches_with_scores(image, stretched_matches.cpu(), deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=False))
stretched_matches_image.save("Visualisations/matches_stretched.png")