import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from util.Affine_Transformations import apply_corotated_strain_with_keypoints, generate_strain_tensors, transform_keypoints, generate_27_strain_tensors
from models import Embedded_Conditional_Residual_MLP, Embedded_Conditional_Fully_Residual_MLP
import time
from matchers.max_similarity import StretcherDualSoftMaxMatcher
from util.matching import draw_matches, draw_matches_with_scores, draw_matching_comparison, print_matching_accuracy
from util.dedode import detect_and_describe, get_affine_deformed_descriptions
from util.image import draw_keypoints

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = get_best_device()
detector = dedode_detector_L(weights = torch.load("models/dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("models/dedode_descriptor_B.pth", map_location = device))
matcher = DualSoftMaxMatcher()
stretcher_matcher = StretcherDualSoftMaxMatcher()
model_dir = "models/stretcher.pth"

# Extract image info
im_path = "data/data/000000000042.jpg"
# im_path = "data/nazim_images/texture1.png"
image = Image.open(im_path)
W, H = image.size
np_image = np.array(image)

if np_image.shape[-1] == 4:
    np_image = np_image[:,:,:3]

# Define parameters
inv_inner_cutoff = 1
double_cutoff = inv_inner_cutoff*2
num_keypoints = 10000
deformation = np.array([-0.5,1.0,0.4])
good_match_threshold = 4
only27 = True

# Extract inner image
uH = (inv_inner_cutoff - 1) * H // double_cutoff
lH = (inv_inner_cutoff + 1) * H // double_cutoff
uW = (inv_inner_cutoff - 1) * W // double_cutoff
lW = (inv_inner_cutoff + 1) * W // double_cutoff
np_inner_image = np_image[uH:lH, uW:lW, :]
h, w = np_inner_image.shape[:2]

# Extract Base Keypoint Descriptions
base_keypoints, base_P, base_descriptions = detect_and_describe(np_inner_image, detector, descriptor, device, num_keypoints)
pixel_keypoints = detector.to_pixel_coords(base_keypoints.cpu(), h, w)
cv2.imwrite("Visualisations/base_keypoints.png", draw_keypoints(np_inner_image, pixel_keypoints[0]))

# Transform Image
whole_pixel_keypoints = pixel_keypoints + torch.tensor([uW, uH])
deformed_image, whole_pixel_deformed_keypoints = apply_corotated_strain_with_keypoints(np_image, whole_pixel_keypoints[0], deformation)
inner_deformed_image = deformed_image[uH:lH, uW:lW, :]
pixel_deformed_base_keypoints = whole_pixel_deformed_keypoints - np.array([uW, uH])
deformed_base_keypoints = detector.to_normalized_coords(torch.tensor(pixel_deformed_base_keypoints), h, w).to(torch.float32)[0]

# Extract Deformed Keypoint Descriptions
deformed_keypoints, deformed_P, deformed_descriptions = detect_and_describe(inner_deformed_image, detector, descriptor, device, num_keypoints)
pixel_deformed_keypoints = detector.to_pixel_coords(deformed_keypoints.cpu(), h, w)
cv2.imwrite("Visualisations/deformed_keypoints.png", draw_keypoints(inner_deformed_image, pixel_deformed_keypoints[0]))

# Match Keypoints     
base_matches, deformed_matches, batch_ids = matcher.match(base_keypoints, base_descriptions,
    deformed_keypoints, deformed_descriptions,
    P_A = base_P, P_B = deformed_P,
    normalize = True, inv_temp=20, threshold = 0.01)#Increasing threshold -> fewer matches, fewer outliers

base_matches, deformed_matches = matcher.to_pixel_coords(base_matches, deformed_matches, h, w, h, w)

print_matching_accuracy(base_matches, deformed_matches, np_image, deformation, uW, uH, threshold = good_match_threshold)

# Image.fromarray(draw_matches(np_inner_image, base_matches.cpu(), inner_deformed_image, deformed_matches.cpu())).save("Visualisations/affine_matches_baseline.png")
baseline_image = Image.fromarray(draw_matches(np_inner_image, base_matches.cpu(), inner_deformed_image, deformed_matches.cpu()))


# Run Model on Base Descriptions
# input_dim = 256
# parameter_dim = 3
# output_dim = 256
# stretcher = Embedded_Conditional_Fully_Residual_MLP(input_dim,parameter_dim,output_dim,hidden_dim=2028,embed_dim=32,num_layers = 6).float().to(device)
# # stretcher = Embedded_Conditional_Residual_MLP(input_dim,parameter_dim,output_dim,hidden_dim=2048,embed_dim=32,num_layers = 6).float().to(device)
# stretcher.load_state_dict(torch.load(model_dir,map_location=device))
# stretcher.eval()
if only27:
    tensors = generate_27_strain_tensors()
else:
    tensors = generate_strain_tensors()
# start = time.time()
# with torch.no_grad():
#     stretched_descriptions = np.array([stretcher(base_descriptions.float(), torch.tensor(tensor).to(torch.float32).to(device).repeat(num_keypoints,1)).cpu() for tensor in tensors])
# stretched_descriptions = torch.tensor(stretched_descriptions).to(device)                
# end = time.time()
# print(f'Time taken for transformation:{end-start}')
stretched_descriptions = get_affine_deformed_descriptions(np_inner_image, pixel_keypoints[0], tensors, detector, descriptor, device)
stretched_descriptions = torch.tensor(stretched_descriptions).to(device)


# Run Matching for Stretched Descriptions
stretched_matches, deformed_matches, batch_ids = stretcher_matcher.match(base_keypoints, stretched_descriptions,
        deformed_keypoints, deformed_descriptions,
        P_A = base_P, P_B = deformed_P,
        normalize = True, inv_temp=20, threshold = 0.01, only27=only27)#Increasing threshold -> fewer matches, fewer outliers

stretched_matches, deformed_matches = stretcher_matcher.to_pixel_coords(stretched_matches, deformed_matches, h, w, h, w)

print_matching_accuracy(stretched_matches, deformed_matches, np_image, deformation, uW, uH, threshold = good_match_threshold)

# Image.fromarray(draw_matches(np_inner_image, stretched_matches.cpu(), inner_deformed_image, deformed_matches.cpu())).save("Visualisations/affine_matches_stretched.png")
stretcher_image = Image.fromarray(draw_matches(np_inner_image, stretched_matches.cpu(), inner_deformed_image, deformed_matches.cpu()))

draw_matching_comparison(baseline_image, stretcher_image, "Visualisations/affine_matches_comparison.png")