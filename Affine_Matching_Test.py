import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import cv2
import numpy as np
import os
from Affine_Transformations import apply_corotated_strain_with_keypoints, generate_strain_tensors
from models import Embedded_Conditional_Residual_MLP
import time

def draw_matches(im_A, kpts_A, im_B, kpts_B):    
    kpts_A = [cv2.KeyPoint(x,y,1.) for x,y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x,y,1.) for x,y in kpts_B.cpu().numpy()]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(len(kpts_A))]
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, 
                    matches_A_to_B, None)
    return ret

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = get_best_device()
detector = dedode_detector_L(weights = torch.load("dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("dedode_descriptor_B.pth", map_location = device))
matcher = DualSoftMaxMatcher()

# Extract image info
im_path = "data/data/000000000042.jpg"
image = Image.open(im_path)
W, H = image.size
np_image = np.array(image)

# Define parameters
inv_inner_cutoff = 2
double_cutoff = inv_inner_cutoff*2
num_keypoints = 100
deformation = np.array([0.5,0.5,0.2])

# Extract inner image
uH = (inv_inner_cutoff - 1) * H // double_cutoff
lH = (inv_inner_cutoff + 1) * H // double_cutoff
uW = (inv_inner_cutoff - 1) * W // double_cutoff
lW = (inv_inner_cutoff + 1) * W // double_cutoff
np_inner_image = np_image[uH:lH, uW:lW, :]
h, w = np_inner_image.shape[:2]

# Extract Base Keypoint Descriptions
np_inner_image = np.array(np_inner_image,dtype=np.uint8)
inner_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(np_inner_image).resize((w,h)))/255.).permute(2,0,1)).float().to(device)[None]
batch = {"image": inner_image}
detections = detector.detect(batch, num_keypoints = num_keypoints)
base_keypoints, base_P = detections["keypoints"], detections["confidence"]
base_descriptions = descriptor.describe_keypoints(batch, base_keypoints.to(device))["descriptions"].squeeze()

# Transform Image
pixel_keypoints = detector.to_pixel_coords(base_keypoints.cpu(), h, w)
whole_pixel_keypoints = pixel_keypoints + torch.tensor([uW, uH])
deformed_image, whole_pixel_deformed_keypoints = apply_corotated_strain_with_keypoints(np_image, whole_pixel_keypoints[0], deformation)
inner_deformed_image = deformed_image[uH:lH, uW:lW, :]
pixel_deformed_keypoints = whole_pixel_deformed_keypoints - np.array([uW, uH])
deformed_base_keypoints = detector.to_normalized_coords(torch.tensor(pixel_deformed_keypoints), h, w).to(torch.float32)[0]
        
# Extract Deformed Keypoint Descriptions
inner_deformed_image = np.array(inner_deformed_image, dtype=np.uint8)
deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(inner_deformed_image).resize((w,h)))/255.).permute(2,0,1)).float().to(device)[None]
deformed_batch = {"image": deformed_image}
deformed_detections = detector.detect(deformed_batch, num_keypoints = num_keypoints)
deformed_keypoints, deformed_P = deformed_detections["keypoints"], deformed_detections["confidence"]
deformed_descriptions = descriptor.describe_keypoints(deformed_batch, deformed_keypoints.to(device))["descriptions"].squeeze()

print(f'Base Keypoints: {base_keypoints}')
print(f'Deformed Keypoints: {deformed_keypoints}')

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
model = Embedded_Conditional_Residual_MLP(input_dim,parameter_dim,output_dim,hidden_dim=512,embed_dim=64,num_layers = 4).float().to(device)
model.load_state_dict(torch.load("models/single_transformation_model.pth",map_location=device))
model.eval()
start = time.time()
with torch.no_grad():
    modeled_descriptions = np.array([model(base_descriptions.float(), torch.tensor(tensor).to(torch.float32).to(device).repeat(num_keypoints,1)).cpu() for tensor in generate_strain_tensors()])
end = time.time()
print(f'Time taken for transformation:{end-start}')
print(f'Modeled Descriptions: {modeled_descriptions.shape}')

# Find Maximum Match per Keypoint


# Run Matching Algorithm
