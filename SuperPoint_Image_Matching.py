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
from lightglue import LightGlue

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = get_best_device()
detector = dedode_detector_L(weights = torch.load("models/dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("models/dedode_descriptor_B.pth", map_location = device))
matcher = DualSoftMaxMatcher()
stretcher_matcher = StretcherDualSoftMaxMatcher()
lightglue_matcher = LightGlue(features='superpoint').eval().to(device)
model_dir = "models/spstretcher_new.pth"

# Extract image info
im_path = 'data/medical_deformed/pl_rest.png'
deformed_im_path = 'data/medical_deformed/pl_def1.png'
# im_path = "data/nazim_images/vms_00000001.png"
# deformed_im_path = "data/nazim_images/vms_00000008.png"
# deformed_im_path = "data/nazim_images/vms_00000021.png"
# im_path = 'data/nazim_images/texture1.png'
# deformed_im_path = 'data/nazim_images/texture2.png'

num_keypoints = 100
stretch_type = 'normal'

hidden_dim = 2048
num_layers = 2
stretcher = TripleNet(256,3,hidden_dim=hidden_dim,num_layers = num_layers).float().to(device)
stretcher.load_state_dict(torch.load(model_dir,map_location=device))
stretcher.eval()

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

if stretch_type == 'only27':
    tensors = generate_27_strain_tensors()
else:
    tensors = generate_strain_tensors()

start = time.time()

# Extract Base Keypoint Descriptions
base_keypoints, base_P, base_descriptions, dense_descriptions, _, base_img_size = sp_detect_and_describe(image, device, num_keypoints)
# baseline_keypoints_vis = draw_keypoints(image, pixel_base_keypoints)
# cv2.imwrite("Visualisations/baseline_keypoints_sp.png", baseline_keypoints_vis)

end = time.time()
base_description_time = end - start
print(f'Base Description Time:{base_description_time}')

start = time.time()

# Extract Deformed Keypoint Descriptions
deformed_keypoints, deformed_P, deformed_descriptions, deformed_dense_descriptions, scales, deformed_img_size = sp_detect_and_describe(deformed_image, device, num_keypoints)
# detected_deformed_keypoints_vis = draw_keypoints(deformed_image, pixel_detected_deformed_keypoints)
# cv2.imwrite("Visualisations/deformed_detected_keypoints_sp.png", detected_deformed_keypoints_vis)

end = time.time()
deformed_description_time = end - start
print(f'Deformed Description Time:{deformed_description_time}')

start = time.time()

# Match Keypoints     
feats0 = {'keypoints': base_keypoints[None].to(device), 'descriptors': base_descriptions[None].to(device), 'image_size': base_img_size[None].to(device)}
feats1 = {'keypoints': deformed_keypoints[None].to(device), 'descriptors': deformed_descriptions[None].to(device), 'image_size': deformed_img_size[None].to(device)}
matches = lightglue_matcher({'image0': feats0, 'image1': feats1})
matches = matches['matches'][0]
base_matches = feats0['keypoints'][0][matches[:,0]]
deformed_matches = feats1['keypoints'][0][matches[:,1]]

end = time.time()
normal_matching_time = end - start
print(f'Normal Matching Time:{normal_matching_time}')

baseline_matches_image = Image.fromarray(draw_matches(image, base_matches.cpu(), deformed_image, deformed_matches.cpu()))
baseline_matches_image.save("Visualisations/matches_baseline.png")

start = time.time()
with torch.no_grad():
    stretched_descriptions = np.array([stretcher(base_descriptions.float().to(device), torch.tensor(tensor).to(torch.float32).to(device).repeat(len(base_descriptions),1)).cpu() for tensor in tensors])
stretched_descriptions = torch.tensor(stretched_descriptions).to(device)                
end = time.time()
stretching_time = end - start
print(f'Stretching Time:{stretching_time}')

start = time.time()
# Run Matching for Stretched Descriptions
stretched_matches, matches_deformed, batch_ids = stretcher_matcher.match(base_keypoints[None].to(device), stretched_descriptions.to(device),
        deformed_keypoints[None].to(device), deformed_descriptions.to(device),
        P_A = base_P, P_B = deformed_P,
        normalize = True, inv_temp=20, threshold = 0.01, stretch_type=stretch_type)#Increasing threshold -> fewer matches, fewer outliers

end = time.time()
stretched_matching_time = end - start
print(f'Stretched Matching Time:{stretched_matching_time}')

stretcher_matches_image = Image.fromarray(draw_matches(image, stretched_matches.cpu(), deformed_image, matches_deformed.cpu()))
stretcher_matches_image.save("Visualisations/matches_stretched.png")

draw_matching_comparison(baseline_matches_image, stretcher_matches_image, "Visualisations/matches_comparison.png")