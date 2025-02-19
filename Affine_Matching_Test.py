import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import numpy as np
import os
from util.Affine_Transformations import apply_corotated_strain_with_keypoints, generate_strain_tensors
from models import Embedded_Conditional_Residual_MLP
import time
from matchers.max_similarity import StretcherDualSoftMaxMatcher
from util.matching import draw_matches
from util.dedode import detect_and_describe

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = get_best_device()
detector = dedode_detector_L(weights = torch.load("models/dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("models/dedode_descriptor_B.pth", map_location = device))
matcher = DualSoftMaxMatcher()
stretcher_matcher = StretcherDualSoftMaxMatcher()

# Extract image info
im_path = "data/data/000000000042.jpg"
image = Image.open(im_path)
W, H = image.size
np_image = np.array(image)

# Define parameters
inv_inner_cutoff = 2
double_cutoff = inv_inner_cutoff*2
num_keypoints = 100
deformation = np.array([0.5,0.5,-0.2])

# Extract inner image
uH = (inv_inner_cutoff - 1) * H // double_cutoff
lH = (inv_inner_cutoff + 1) * H // double_cutoff
uW = (inv_inner_cutoff - 1) * W // double_cutoff
lW = (inv_inner_cutoff + 1) * W // double_cutoff
np_inner_image = np_image[uH:lH, uW:lW, :]
h, w = np_inner_image.shape[:2]

# Extract Base Keypoint Descriptions
base_keypoints, base_P, base_descriptions = detect_and_describe(np_inner_image, detector, descriptor, device, num_keypoints)

# Transform Image
pixel_keypoints = detector.to_pixel_coords(base_keypoints.cpu(), h, w)
whole_pixel_keypoints = pixel_keypoints + torch.tensor([uW, uH])
deformed_image, whole_pixel_deformed_keypoints = apply_corotated_strain_with_keypoints(np_image, whole_pixel_keypoints[0], deformation)
inner_deformed_image = deformed_image[uH:lH, uW:lW, :]
pixel_deformed_keypoints = whole_pixel_deformed_keypoints - np.array([uW, uH])
deformed_base_keypoints = detector.to_normalized_coords(torch.tensor(pixel_deformed_keypoints), h, w).to(torch.float32)[0]
        
# Extract Deformed Keypoint Descriptions
deformed_keypoints, deformed_P, deformed_descriptions = detect_and_describe(inner_deformed_image, detector, descriptor, device, num_keypoints)

# Match Keypoints     
matches_base, matches_deformed, batch_ids = matcher.match(base_keypoints, base_descriptions,
    deformed_keypoints, deformed_descriptions,
    P_A = base_P, P_B = deformed_P,
    normalize = True, inv_temp=20, threshold = 0.01)#Increasing threshold -> fewer matches, fewer outliers

matches_base, matches_deformed = matcher.to_pixel_coords(matches_base, matches_deformed, h, w, h, w)

Image.fromarray(draw_matches(np_inner_image, matches_base.cpu(), inner_deformed_image, matches_deformed.cpu())).save("Visualisations/affine_matches_baseline.png")

# Run Model on Base Descriptions
input_dim = 256
parameter_dim = 3
output_dim = 256
stretcher = Embedded_Conditional_Residual_MLP(input_dim,parameter_dim,output_dim,hidden_dim=1024,embed_dim=32,num_layers = 5).float().to(device)
stretcher.load_state_dict(torch.load("models/stretcher.pth",map_location=device))
stretcher.eval()
start = time.time()
with torch.no_grad():
    stretched_descriptions = np.array([stretcher(base_descriptions.float(), torch.tensor(tensor).to(torch.float32).to(device).repeat(num_keypoints,1)).cpu() for tensor in generate_strain_tensors()])
stretched_descriptions = torch.tensor(stretched_descriptions).to(device)                
end = time.time()
print(f'Time taken for transformation:{end-start}')

# Run Matching for Stretched Descriptions
stretched_matches, matches_deformed, batch_ids = stretcher_matcher.match(base_keypoints, stretched_descriptions,
        deformed_keypoints, deformed_descriptions,
        P_A = base_P, P_B = deformed_P,
        normalize = False, inv_temp=20, threshold = 0.01)#Increasing threshold -> fewer matches, fewer outliers

stretched_matches, matches_deformed = stretcher_matcher.to_pixel_coords(stretched_matches, matches_deformed, h, w, h, w)

Image.fromarray(draw_matches(np_inner_image, stretched_matches.cpu(), inner_deformed_image, matches_deformed.cpu())).save("Visualisations/affine_matches_stretched.png")