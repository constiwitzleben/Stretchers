from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import numpy as np
from lightglue import LightGlue, DISK, ALIKED, SuperPoint
from lightglue.utils import load_image, rbd
from fenics_2D_elasticity_grid_proj import create_deformed_medical_image_pair, track_pixel_displacement, get_strain
from util.matching import draw_matches_with_scores, strain_entropy, strain_balanced_precision
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="__array_wrap__ must accept context and return_scalar")

good_match_threshold = 5

device = device = get_best_device()

# pretty_im_path = 'data/pretty_data/rest.png'
# pretty_deformed_im_path = 'data/pretty_data/deformed.png'
pretty_im_path = 'data/medical_deformed/pig_liver_to_elongate.png'
pretty_deformed_im_path = 'data/medical_deformed/pig_liver_to_elongate_deformed.png'
im_path = 'data/medical_deformed/pig_liver_to_elongate.png'
deformed_im_path = 'data/medical_deformed/pig_liver_to_elongate_deformed.png'

# deformations = np.array([[8e6, 1e6], [8e6, -1e6], [-4e6, 2e6], [-4e6, -2e6]])

u, s, new_lx, new_ly, bottom_left = create_deformed_medical_image_pair(im_path, deformed_im_path, 8e6, -1e6 )

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

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly, bottom_left) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good_mask = distances < good_match_threshold
num_good_matches = good_mask.sum().item()

num_matches = len(distances)
precision = num_good_matches / num_matches
matching_score = num_good_matches / min(len(feats0['keypoints']), len(feats1['keypoints']))
entropy = strain_entropy(s, base_matches[good_mask].cpu(), W, H)
sb_precision = strain_balanced_precision(s, base_matches.cpu(), good_mask, W, H)

print(f'DISK LG Precision: {precision} ({num_good_matches} / {num_matches})')
print(f'DISK LG Matching Score: {matching_score}')
print(f'DISK LG Number of Matches: {num_matches}')
print(f'DISK LG Entropy: {entropy}')
print(f'DISK LG Strain Balanced Precision: {sb_precision}')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_disk_lg.png')

#--------------------------------------------------------------------------------------------------------------------------

base_keypoints = feats0['keypoints'][None]
base_descriptions = feats0['descriptors'][None]
base_P = None
deformed_keypoints = feats1['keypoints'][None]
deformed_descriptions = feats1['descriptors'][None]
deformed_P = None

# print(f'Number of base keypoints detected by DISK: {base_keypoints.shape}')
# print(f'Number of deformed keypoints detected by DISK: {deformed_keypoints.shape}')

base_matches, deformed_matches, batch_ids = dsm_matcher.match(base_keypoints.to(device), base_descriptions.to(device),
            deformed_keypoints.to(device), deformed_descriptions.to(device),
            P_A = base_P, P_B = deformed_P,
            normalize = True, inv_temp=20, threshold = 0.01)

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly, bottom_left) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
num_good_matches = (distances < good_match_threshold).sum().item()

num_matches = len(distances)
precision = num_good_matches / num_matches
matching_score = num_good_matches / min(base_keypoints.shape[1], deformed_keypoints.shape[1])
entropy = strain_entropy(s, base_matches.cpu(), W, H)

print(f'DISK DSM Precision: {precision} ({num_good_matches} / {num_matches})')
print(f'DISK DSM Matching Score: {matching_score}')
print(f'DISK DSM Number of Matches: {num_matches}')
print(f'DISK DSM Entropy: {entropy}')

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

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly, bottom_left) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
num_good_matches = (distances < good_match_threshold).sum().item()

num_matches = len(distances)
precision = num_good_matches / num_matches
matching_score = num_good_matches / min(len(feats0['keypoints']), len(feats1['keypoints']))
entropy = strain_entropy(s, base_matches.cpu(), W, H)

print(f'ALIKED LG Precision: {precision} ({num_good_matches} / {num_matches})')
print(f'ALIKED LG Matching Score: {matching_score}')
print(f'ALIKED LG Number of Matches: {num_matches}')
print(f'ALIKED LG Entropy: {entropy}')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_aliked_lg.png')

#--------------------------------------------------------------------------------------------------------------------------

base_keypoints = feats0['keypoints'][None]
base_descriptions = feats0['descriptors'][None]
base_P = None
deformed_keypoints = feats1['keypoints'][None]
deformed_descriptions = feats1['descriptors'][None]
deformed_P = None

# print(f'Number of base keypoints detected by Aliked: {base_keypoints.shape}')
# print(f'Number of deformed keypoints detected by Aliked: {deformed_keypoints.shape}')

base_matches, deformed_matches, batch_ids = dsm_matcher.match(base_keypoints.to(device), base_descriptions.to(device),
            deformed_keypoints.to(device), deformed_descriptions.to(device),
            P_A = base_P, P_B = deformed_P,
            normalize = True, inv_temp=20, threshold = 0.01)

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly, bottom_left) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
num_good_matches = (distances < good_match_threshold).sum().item()

num_matches = len(distances)
precision = num_good_matches / num_matches
matching_score = num_good_matches / min(base_keypoints.shape[1], deformed_keypoints.shape[1])
entropy = strain_entropy(s, base_matches.cpu(), W, H)

print(f'ALIKED DSM Precision: {precision} ({num_good_matches} / {num_matches})')
print(f'ALIKED DSM Matching Score: {matching_score}')
print(f'ALIKED DSM Number of Matches: {num_matches}')
print(f'ALIKED DSM Entropy: {entropy}')

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

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly, bottom_left) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
num_good_matches = (distances < good_match_threshold).sum().item()

num_matches = len(distances)
precision = num_good_matches / num_matches
matching_score = num_good_matches / min(len(feats0['keypoints']), len(feats1['keypoints']))
entropy = strain_entropy(s, base_matches.cpu(), W, H)

print(f'SP LG Precision: {precision} ({num_good_matches} / {num_matches})')
print(f'SP LG Matching Score: {matching_score}')
print(f'SP LG Number of Matches: {num_matches}')
print(f'SP LG Entropy: {entropy}')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_SP_lg.png')

#--------------------------------------------------------------------------------------------------------------------------

base_keypoints = feats0['keypoints'][None]
base_descriptions = feats0['descriptors'][None]
base_P = None
deformed_keypoints = feats1['keypoints'][None]
deformed_descriptions = feats1['descriptors'][None]
deformed_P = None

# print(f'Number of base keypoints detected by SP: {base_keypoints.shape}')
# print(f'Number of deformed keypoints detected by SP: {deformed_keypoints.shape}')

base_matches, deformed_matches, batch_ids = dsm_matcher.match(base_keypoints.to(device), base_descriptions.to(device),
            deformed_keypoints.to(device), deformed_descriptions.to(device),
            P_A = base_P, P_B = deformed_P,
            normalize = True, inv_temp=20, threshold = 0.01)

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly, bottom_left) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
num_good_matches = (distances < good_match_threshold).sum().item()

num_matches = len(distances)
precision = num_good_matches / num_matches
matching_score = num_good_matches / min(base_keypoints.shape[1], deformed_keypoints.shape[1])
entropy = strain_entropy(s, base_matches.cpu(), W, H)

print(f'SP DSM Precision: {precision} ({num_good_matches} / {num_matches})')
print(f'SP DSM Matching Score: {matching_score}')
print(f'SP DSM Number of Matches: {num_matches}')
print(f'SP DSM Entropy: {entropy}')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
baseline_matches_image.save('Visualisations/matches_SP_dsm.png')

#--------------------------------------------------------------------------------------------------------------------------