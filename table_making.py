import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import numpy as np
import os
from util.Affine_Transformations import apply_corotated_strain_with_keypoints, generate_strain_tensors, generate_27_strain_tensors
from models import Embedded_Conditional_Residual_MLP, Embedded_Conditional_Fully_Residual_MLP, TripleNet
import time
from matchers.max_similarity import StretcherDualSoftMaxMatcher
from util.matching import draw_matches, draw_matching_comparison
from util.superpoint import sp_detect_and_describe
from util.image import draw_keypoints
from lightglue import LightGlue, DISK, ALIKED, SuperPoint
from lightglue.utils import load_image, rbd
from fenics_2D_elasticity_grid_proj import create_deformed_medical_image_pair, track_pixel_displacement
from util.matching import draw_matches_with_scores

good_match_threshold = 5

device = device = get_best_device()

pretty_im_path = 'data/pretty_data/rest.png'
pretty_deformed_im_path = 'data/pretty_data/deformed.png'
im_path = 'data/medical_deformed/pig_liver_to_elongate.png'
deformed_im_path = 'data/medical_deformed/pig_liver_to_elongate_deformed.png'

# deformations = np.array([[8e6, 1e6], [1e6, 8e6], [8e6, -1e6], [1e6, -8e6]])

u, s, new_lx, new_ly = create_deformed_medical_image_pair(im_path, deformed_im_path, 8e6, 1e6 )

image = Image.open(im_path)
deformed_image = Image.open(deformed_im_path)
W, H = image.size
dW, dH = deformed_image.size
image = np.array(image)
if image.shape[-1] == 4:
    image = image[:,:,:3]
deformed_image = np.array(deformed_image)
if deformed_image.shape[-1] == 4:
    deformed_image = deformed_image[:,:,:3]

pretty_image = Image.open(pretty_im_path)
pretty_deformed_image = Image.open(pretty_deformed_im_path)
pretty_image = np.array(pretty_image)
if pretty_image.shape[-1] == 4:
    pretty_image = pretty_image[:,:,:3]
pretty_deformed_image = np.array(pretty_deformed_image)
if pretty_deformed_image.shape[-1] == 4:
    pretty_deformed_image = pretty_deformed_image[:,:,:3]

#--------------------------------------------------------------------------------------------------------------------------

extractor = DISK(max_num_keypoints=2048).eval().to(device)
lg_matcher = LightGlue(features='disk').eval().to(device)
dsm_matcher = DualSoftMaxMatcher()

image0 = load_image(im_path).to(device)
image1 = load_image(deformed_im_path).to(device)
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

matches01 = lg_matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
base_matches = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
deformed_matches = feats1['keypoints'][matches[..., 1]]

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'DISK LG Accuracy: {accuracy} ({good} / {total})')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_disk_lg.png')

#--------------------------------------------------------------------------------------------------------------------------

base_keypoints = feats0['keypoints'][None]
base_descriptions = feats0['descriptors'][None]
base_P = None
deformed_keypoints = feats1['keypoints'][None]
deformed_descriptions = feats1['descriptors'][None]
deformed_P = None

print(f'Number of base keypoints detected by DISK: {base_keypoints.shape}')
print(f'Number of deformed keypoints detected by DISK: {deformed_keypoints.shape}')

base_matches, deformed_matches, batch_ids = dsm_matcher.match(base_keypoints.to(device), base_descriptions.to(device),
            deformed_keypoints.to(device), deformed_descriptions.to(device),
            P_A = base_P, P_B = deformed_P,
            normalize = True, inv_temp=20, threshold = 0.01)

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'DISK DSM Accuracy: {accuracy} ({good} / {total})')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_disk_dsm.png')

#---------------------------------------------------------------------------------------------------------------------------------

extractor = ALIKED(max_num_keypoints=2048).eval().to(device)
lg_matcher = LightGlue(features='aliked').eval().to(device)
dsm_matcher = DualSoftMaxMatcher()

image0 = load_image(im_path).to(device)
image1 = load_image(deformed_im_path).to(device)
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

matches01 = lg_matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
base_matches = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
deformed_matches = feats1['keypoints'][matches[..., 1]]

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'ALIKED LG Accuracy: {accuracy} ({good} / {total})')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_aliked_lg.png')

#--------------------------------------------------------------------------------------------------------------------------

base_keypoints = feats0['keypoints'][None]
base_descriptions = feats0['descriptors'][None]
base_P = None
deformed_keypoints = feats1['keypoints'][None]
deformed_descriptions = feats1['descriptors'][None]
deformed_P = None

print(f'Number of base keypoints detected by Aliked: {base_keypoints.shape}')
print(f'Number of deformed keypoints detected by Aliked: {deformed_keypoints.shape}')

base_matches, deformed_matches, batch_ids = dsm_matcher.match(base_keypoints.to(device), base_descriptions.to(device),
            deformed_keypoints.to(device), deformed_descriptions.to(device),
            P_A = base_P, P_B = deformed_P,
            normalize = True, inv_temp=20, threshold = 0.01)

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'ALIKED DSM Accuracy: {accuracy} ({good} / {total})')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_aliked_dsm.png')

#---------------------------------------------------------------------------------------------------------------------------------

extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
lg_matcher = LightGlue(features='superpoint').eval().to(device)
dsm_matcher = DualSoftMaxMatcher()

image0 = load_image(im_path).to(device)
image1 = load_image(deformed_im_path).to(device)
feats0 = extractor.extract(image0)
feats1 = extractor.extract(image1)

matches01 = lg_matcher({'image0': feats0, 'image1': feats1})
feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
matches = matches01['matches']  # indices with shape (K,2)
base_matches = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
deformed_matches = feats1['keypoints'][matches[..., 1]]

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'SP LG Accuracy: {accuracy} ({good} / {total})')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_SP_lg.png')

#--------------------------------------------------------------------------------------------------------------------------

base_keypoints = feats0['keypoints'][None]
base_descriptions = feats0['descriptors'][None]
base_P = None
deformed_keypoints = feats1['keypoints'][None]
deformed_descriptions = feats1['descriptors'][None]
deformed_P = None

print(f'Number of base keypoints detected by SP: {base_keypoints.shape}')
print(f'Number of deformed keypoints detected by SP: {deformed_keypoints.shape}')

base_matches, deformed_matches, batch_ids = dsm_matcher.match(base_keypoints.to(device), base_descriptions.to(device),
            deformed_keypoints.to(device), deformed_descriptions.to(device),
            P_A = base_P, P_B = deformed_P,
            normalize = True, inv_temp=20, threshold = 0.01)

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'SP DSM Accuracy: {accuracy} ({good} / {total})')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_SP_dsm.png')

#--------------------------------------------------------------------------------------------------------------------------