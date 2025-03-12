import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image, ImageOps
import numpy as np
import os
from util.Affine_Transformations import apply_corotated_strain_with_keypoints, generate_strain_tensors, generate_27_strain_tensors, generate_larger_strain_tensors, generate_test_strain_tensors
from models import Embedded_Conditional_Residual_MLP, Embedded_Conditional_Fully_Residual_MLP, SuperNet, TripleNet
import time
from matchers.max_similarity import StretcherDualSoftMaxMatcher
from util.matching import draw_matches, draw_matching_comparison, draw_matches_with_scores
from util.dedode import detect_and_describe, get_affine_deformed_descriptions
from util.image import draw_keypoints, visualize_keypoint_similarities, crop_to_square
from util.superpoint import sp_detect_and_describe, custom_sample_descriptors, sp_get_affine_deformed_descriptions
from fenics_2D_elasticity_grid_proj import create_deformed_medical_image_pair, track_pixel_displacement
import math
from lightglue import LightGlue
from lightglue.utils import rbd


device = get_best_device()
detector = dedode_detector_L(weights = torch.load("models/dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("models/dedode_descriptor_B.pth", map_location = device))
matcher = DualSoftMaxMatcher()
stretcher_matcher = StretcherDualSoftMaxMatcher()
lightglue_matcher = LightGlue(features='superpoint').eval().to(device)

num_keypoints = 10000
stretch_type = 'normal'
good_match_threshold = 5
chosen_keypoints = None
model = 'superpoint'
base_matching = 'lightglue'
affine_matching = 'lightglue'
model_dir = "models/spstretcher_new.pth"
stretching_type = 'free'

if stretch_type == 'only27':
    tensors = np.array(generate_27_strain_tensors())
elif stretch_type == 'larger':
    tensors = np.array(generate_larger_strain_tensors())
elif stretch_type == 'test':
    tensors = np.array(generate_test_strain_tensors())
else:
    tensors = np.array(generate_strain_tensors())

# im_path = 'data/medical_deformed/pig_liver_rest_480_square.png'
# im_path_to_deform = 'data/medical_deformed/pig_liver_to_elongate_480_square.png'
# deformed_im_path = 'data/medical_deformed/pig_liver_elongated_480.png'
im_path = 'data/medical_deformed/pig_liver_to_elongate.png'
deformed_im_path = 'data/medical_deformed/pig_liver_to_elongate_deformed.png'
# im_path = 'data/medical_deformed/skull.png'
# deformed_im_path = 'data/medical_deformed/skull_deformed.png'
# im_path = 'data/medical_testing_data/c_104_v_2959_f_377.jpg'
# deformed_im_path = 'data/deformed_stuff/c_104_v_2959_f_377.png'
# im_path = 'data/medical_training_data/c_185_v_55393_f_77.jpg'
# deformed_im_path = 'data/deformed_stuff/c_185_v_55393_f_77.png'
pretty_im_path = 'data/pretty_data/rest.png'
pretty_deformed_im_path = 'data/pretty_data/deformed.png'

hidden_dim = 2048
num_layers = 2
stretcher = TripleNet(256,3,hidden_dim=hidden_dim,num_layers = num_layers).float().to(device)
stretcher.load_state_dict(torch.load(model_dir,map_location=device))
stretcher.eval()

u, new_lx, new_ly = create_deformed_medical_image_pair(im_path, deformed_im_path, 8e6, 1e6 )

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
pretty_image = Image.open(pretty_im_path)
pretty_deformed_image = Image.open(pretty_deformed_im_path)
pretty_image = np.array(pretty_image, dtype=np.uint8)
if pretty_image.shape[-1] == 4:
    pretty_image = pretty_image[:,:,:3]
pretty_deformed_image = np.array(pretty_deformed_image, dtype=np.uint8)
if pretty_deformed_image.shape[-1] == 4:
    pretty_deformed_image = pretty_deformed_image[:,:,:3]

#----------------------------------------------------------------------------------
#   Extract Base Keypoints and Descriptions
#----------------------------------------------------------------------------------

if model == 'dedode':
    base_keypoints, base_P, base_descriptions = detect_and_describe(image, detector, descriptor, device, num_keypoints)
    pixel_base_keypoints = detector.to_pixel_coords(base_keypoints.cpu(), H, W)[0]
    baseline_keypoints_vis = draw_keypoints(image, pixel_base_keypoints)
    cv2.imwrite("Visualisations/baseline_keypoints_dedode.png", baseline_keypoints_vis)
elif model == 'superpoint':
    pixel_base_keypoints, base_P, base_descriptions, dense_descriptions, _, base_img_size = sp_detect_and_describe(image, device, num_keypoints)
    baseline_keypoints_vis = draw_keypoints(pretty_image, pixel_base_keypoints)
    cv2.imwrite("Visualisations/baseline_keypoints_sp_dsm.png", baseline_keypoints_vis)

#----------------------------------------------------------------------------------
#   Find Base Keypoints in Deformed Image and Extract Deformed Keypoint Descriptions
#----------------------------------------------------------------------------------

pixel_deformed_keypoints = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in pixel_base_keypoints.cpu()])[None]
deformed_keypoints = detector.to_normalized_coords(torch.tensor(pixel_deformed_keypoints), dH, dW).to(torch.float32)

# Extract Deformed Keypoint Descriptions
if  model == 'dedode':
    deformed_keypoints_vis = draw_keypoints(deformed_image, pixel_deformed_keypoints[0])
    cv2.imwrite("Visualisations/deformed_keypoints_dedode.png", deformed_keypoints_vis)
    im = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(deformed_image).resize((dW,dH)))/255.).permute(2,0,1)).float().to(device)[None]
    batch = {"image": im}
    deformed_descriptions = descriptor.describe_keypoints(batch, deformed_keypoints.to(device))['descriptions'].squeeze()
elif model == 'superpoint':
    deformed_keypoints_vis = draw_keypoints(pretty_deformed_image, pixel_deformed_keypoints[0])
    cv2.imwrite("Visualisations/deformed_keypoints_sp_dsm.png", deformed_keypoints_vis)
    _, _, _, deformed_dense_descriptions, scales, _ = sp_detect_and_describe(deformed_image, device, num_keypoints)
    sp_keypoints = (torch.tensor(pixel_deformed_keypoints[0]) + 0.5) * scales - 0.5
    deformed_descriptions = custom_sample_descriptors(sp_keypoints, deformed_dense_descriptions.cpu()).permute(0,2,1)[0]

#----------------------------------------------------------------------------------
#   Plot similarity between base and deformed descriptions
#----------------------------------------------------------------------------------

normed1 = torch.nn.functional.normalize(base_descriptions, p=2, dim=1)
normed2 = torch.nn.functional.normalize(deformed_descriptions, p=2, dim=1)
similarity_matrix = torch.mm(normed1, normed2.T)
# print(similarity_matrix)
cosines = torch.diag(similarity_matrix)

deformed_image_vis, min_cos, max_cos = visualize_keypoint_similarities(pretty_deformed_image, pixel_deformed_keypoints[0], cosines)

if model == 'dedode':
    cv2.imwrite("Visualisations/baseline_deformed_keypoints_similarity_dedode.png", deformed_image_vis)
elif model == 'superpoint':
    cv2.imwrite("Visualisations/baseline_deformed_keypoints_similarity_sp_dsm.png", deformed_image_vis)

#----------------------------------------------------------------------------------
#  Find similarity between affine transformed and deformed descriptions
#----------------------------------------------------------------------------------

if model == 'dedode':
    affined_descriptions = torch.tensor(get_affine_deformed_descriptions(image, pixel_base_keypoints, tensors, detector, descriptor, device)).to(device)
elif model == 'superpoint':
    affined_descriptions = sp_get_affine_deformed_descriptions(image, pixel_base_keypoints, tensors, device).to(torch.float32).to(device)
    print(affined_descriptions.shape)

normed1 = torch.nn.functional.normalize(affined_descriptions.cpu(), p=2, dim=2)
normed2 = torch.nn.functional.normalize(deformed_descriptions.cpu(), p=2, dim=1)
max_similarity_matrix = torch.full((len(deformed_descriptions), len(deformed_descriptions)), float('-inf'), device=normed1.device)
max_index_matrix = torch.full((len(deformed_descriptions), len(deformed_descriptions)),-1,dtype=torch.long, device=normed1.device)

# Loop over batches
for i in range(normed1.shape[0]):  # Iterate over 125 batches
    sim_matrix = torch.matmul(normed1[i], normed2.T)  # Shape: (10000, 10000)
    max_similarity_matrix = torch.maximum(max_similarity_matrix, sim_matrix)  # Element-wise max update
    update_mask = sim_matrix > max_similarity_matrix
    max_index_matrix[update_mask] = i

# print(similarity_matrix)
cosines = torch.diag(max_similarity_matrix)
max_tensors = torch.diag(max_index_matrix)

affined_deformed_image_vis, _, _ = visualize_keypoint_similarities(pretty_deformed_image, pixel_deformed_keypoints[0], cosines.cpu(), min_cos.cpu(), max_cos.cpu())

# Save and display
if model == 'dedode':
    cv2.imwrite("Visualisations/affined_deformed_keypoints_similarity_dedode.png", affined_deformed_image_vis)
elif model == 'superpoint':
    cv2.imwrite("Visualisations/affined_deformed_keypoints_similarity_sp_dsm.png", affined_deformed_image_vis)

#----------------------------------------------------------------------------------
#  Find similarity between stretched and deformed descriptions
#----------------------------------------------------------------------------------

# Run Model on Base Descriptions

if stretching_type == 'free':
    print('Starting Stretching')
    with torch.no_grad():
        stretched_descriptions = np.array([stretcher(base_descriptions.to(torch.float32).to(device), torch.tensor(tensor).to(torch.float32).to(device).repeat(len(base_descriptions),1)).cpu() for tensor in tensors])
    stretched_descriptions = torch.tensor(stretched_descriptions).to(device)
    print(stretched_descriptions.shape)   

    normed1 = torch.nn.functional.normalize(stretched_descriptions.cpu(), p=2, dim=2)
    normed2 = torch.nn.functional.normalize(deformed_descriptions.cpu(), p=2, dim=1)
    max_similarity_matrix = torch.full((len(deformed_descriptions), len(deformed_descriptions)), float('-inf'), device=normed1.device)

    # Loop over batches
    for i in range(normed1.shape[0]):  # Iterate over 125 batches
        sim_matrix = torch.matmul(normed1[i], normed2.T)  # Shape: (10000, 10000)
        max_similarity_matrix = torch.maximum(max_similarity_matrix, sim_matrix)  # Element-wise max update

    cosines = torch.diag(max_similarity_matrix)

elif stretching_type == 'tracked':
    with torch.no_grad():
        stretched_descriptions = stretcher(base_descriptions.to(torch.float32).to(device), torch.tensor(tensors[max_tensors]).to(torch.float32).to(device)).cpu()

    normed1 = torch.nn.functional.normalize(stretched_descriptions, p=2, dim=1)
    normed2 = torch.nn.functional.normalize(deformed_descriptions, p=2, dim=1)
    similarity_matrix = torch.mm(normed1, normed2.T)

    cosines = torch.diag(similarity_matrix)

stretched_deformed_image_vis, _, _ = visualize_keypoint_similarities(pretty_deformed_image, pixel_deformed_keypoints[0], cosines.cpu(), min_cos.cpu(), max_cos.cpu())

# Save and display
if model == 'dedode':
    cv2.imwrite("Visualisations/stretched_deformed_keypoints_similarity_dedode.png", stretched_deformed_image_vis)
elif model == 'superpoint':
    cv2.imwrite("Visualisations/stretched_deformed_keypoints_similarity_sp_dsm.png", stretched_deformed_image_vis)

#----------------------------------------------------------------------------------
#  Find similarity between affined and stretched descriptions
#----------------------------------------------------------------------------------

print(max_tensors.shape)
print(affined_descriptions[max_tensors].shape)
# Run Model on Base Descriptions
if stretching_type == 'free':
    keypoint_indices = torch.arange(max_tensors.shape[0])
    normed1 = torch.nn.functional.normalize(affined_descriptions[max_tensors,keypoint_indices,:].cpu(), p=2, dim=1)
    print(normed1.shape)
    normed2 = torch.nn.functional.normalize(stretched_descriptions[max_tensors,keypoint_indices,:].cpu(), p=2, dim=1)
    print(normed2.shape)
    similarity_matrix = torch.mm(normed1, normed2.T)

elif stretching_type == 'tracked':
    normed1 = torch.nn.functional.normalize(affined_descriptions[max_tensors].cpu(), p=2, dim=1)
    normed2 = torch.nn.functional.normalize(stretched_descriptions.cpu(), p=2, dim=1)
    similarity_matrix = torch.mm(normed1, normed2.T)

# print(similarity_matrix)
cosines = torch.diag(similarity_matrix)

affined_stretched_image_vis, _, _ = visualize_keypoint_similarities(pretty_deformed_image, pixel_deformed_keypoints[0], cosines.cpu(), min_cos.cpu(), max_cos.cpu())

# Save and display
if model == 'dedode':
    cv2.imwrite("Visualisations/affined_stretched_keypoints_similarity_dedode.png", affined_stretched_image_vis)
elif model == 'superpoint':
    cv2.imwrite("Visualisations/affined_stretched_keypoints_similarity_sp_dsm.png", affined_stretched_image_vis)

#----------------------------------------------------------------------------------
#   Matching Baseline to Deformed
#---------------------------------------------------------------------------------- 

if model == 'dedode':
    detected_deformed_keypoints, deformed_P, detected_deformed_descriptions = detect_and_describe(deformed_image, detector, descriptor, device, num_keypoints)
    pixel_detected_deformed_keypoints = detector.to_pixel_coords(detected_deformed_keypoints.cpu(), dH, dW)[0]
if model == 'superpoint':
    pixel_detected_deformed_keypoints, deformed_P, detected_deformed_descriptions, detected_deformed_dense_descriptions, scales, deformed_img_size = sp_detect_and_describe(deformed_image, device, num_keypoints)
    detected_deformed_keypoints_vis = draw_keypoints(pretty_deformed_image, pixel_detected_deformed_keypoints)
    cv2.imwrite("Visualisations/deformed_detected_keypoints_sp_dsm.png", detected_deformed_keypoints_vis)
    print(f'Number of deformed keypoints: {len(pixel_detected_deformed_keypoints)}')

if base_matching == 'dualsoftmax':
    base_matches, deformed_matches, batch_ids = matcher.match(pixel_base_keypoints[None].to(device), base_descriptions.to(device),
            pixel_detected_deformed_keypoints[None].to(device), detected_deformed_descriptions.to(device),
            P_A = base_P, P_B = deformed_P,
            normalize = True, inv_temp=20, threshold = 0.01)
elif base_matching == 'lightglue':
    feats0 = {'keypoints': pixel_base_keypoints[None].to(device), 'descriptors': base_descriptions[None].to(device), 'image_size': base_img_size[None].to(device)}
    feats1 = {'keypoints': pixel_detected_deformed_keypoints[None].to(device), 'descriptors': detected_deformed_descriptions[None].to(device), 'image_size': deformed_img_size[None].to(device)}

    affine_matches = lightglue_matcher({'image0': feats0, 'image1': feats1})

    matches = affine_matches['matches'][0]
    base_matches = feats0['keypoints'][0][matches[:,0]]
    deformed_matches = feats1['keypoints'][0][matches[:,1]]

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in base_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'Baseline Accuracy: {accuracy} ({good} / {total})')

baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
if model == 'dedode':
    baseline_matches_image.save("Visualisations/matches_baseline_dedode.png")
elif model == 'superpoint':
    baseline_matches_image.save("Visualisations/matches_baseline_sp_dsm.png")

# #----------------------------------------------------------------------------------
# #   Matching Affine Transformed to Deformed
# #----------------------------------------------------------------------------------

# if affine_matching == 'dualsoftmax':
#     affined_matches, deformed_matches, batch_ids = stretcher_matcher.match(pixel_base_keypoints[None].to(device), affined_descriptions.to(device),
#             pixel_detected_deformed_keypoints[None].to(device), detected_deformed_descriptions.to(device),
#             P_A = base_P, P_B = deformed_P,
#             normalize = True, inv_temp=20, threshold = 0.05, stretch_type=stretch_type)
# elif affine_matching == 'lightglue':

# # ----------------------------------------------------------------------
#     # New Version
#     best_matches = {}
#     base_keypoint_carrier = {}
#     best_scores = {}
#     best_descriptor_versions = {}

#     for i in range(affined_descriptions.shape[0]):  # Iterate over batch

#         feats0 = {
#             'keypoints': pixel_base_keypoints[None].to(device),
#             'descriptors': affined_descriptions[i][None].to(device),
#             'image_size': base_img_size[None].to(device)
#         }
#         feats1 = {
#             'keypoints': pixel_detected_deformed_keypoints[None].to(device),
#             'descriptors': detected_deformed_descriptions[None].to(device),
#             'image_size': deformed_img_size[None].to(device)
#         }

#         affine_matches = lightglue_matcher({'image0': feats0, 'image1': feats1})

#         matches = affine_matches['matches'][0]
#         scores = affine_matches['scores'][0]  # Assuming LightGlue provides scores

#         # New Version
#         base_indices = matches[:, 0]  # Indices of matched keypoints in the base image
#         base_keypoints = feats0['keypoints'][0][matches[:,0]]  # Corresponding matched keypoints
#         deformed_keypoints = feats1['keypoints'][0][matches[:, 1]]  # Corresponding matched keypoints

#         for j in range(len(base_indices)):
#             keypoint_idx = base_indices[j].item()  # Convert to int for dictionary key
#             score = scores[j].item()

#             # If keypoint is not stored yet OR this match has a higher score, update it
#             if keypoint_idx not in best_matches or score > best_scores[keypoint_idx]:
#                 best_matches[keypoint_idx] = deformed_keypoints[j]  # Save best-matching keypoint
#                 base_keypoint_carrier[keypoint_idx] = base_keypoints[j]  # Save corresponding base keypoint
#                 best_scores[keypoint_idx] = score  # Save best score
#                 best_descriptor_versions[keypoint_idx] = i  # Track descriptor version

#     # Convert to tensors
#     unique_base_matches = torch.stack(list(base_keypoint_carrier.values()))  # Base keypoints
#     # unique_base_matches = torch.tensor(list(best_matches.keys()), device=device)  # Indices of best keypoints
#     unique_deformed_matches = torch.stack(list(best_matches.values()))  # Matched keypoints
#     unique_scores = torch.tensor(list(best_scores.values()), device=device)  # Best scores
#     descriptor_versions = torch.tensor(list(best_descriptor_versions.values()), device=device)  # Descriptor versions

#     # Select top 500 unique matches based on score
#     top_indices = unique_scores.argsort(descending=True)[:200]

#     affined_matches = unique_base_matches[top_indices]  # Best 500 keypoints (indices)
#     print(affined_matches)
#     print(affined_matches.shape)
#     deformed_matches = unique_deformed_matches[top_indices]  # Best 500 matched keypoints
#     print(deformed_matches)
#     print(deformed_matches.shape)
#     matched_transformations = descriptor_versions[top_indices]  # Descriptor versions for these matches
#     print(matched_transformations)
#     print(matched_transformations.shape)
#     # -------------------------------------------------------------------------------------------------------------------------

#     # # Number of descriptor versions
#     # num_versions = affined_descriptions.shape[0]
#     # num_keypoints = pixel_base_keypoints.shape[0]

#     # # Preallocate tensors for storing the best matches
#     # best_scores = torch.full((num_keypoints,), float('-inf'), device=device)  # Store highest score per keypoint
#     # best_matches = torch.zeros((num_keypoints, 2), device=device)  # Store best deformed keypoints
#     # base_keypoint_carrier = torch.zeros((num_keypoints, 2), device=device)  # Store base keypoints
#     # best_descriptor_versions = torch.full((num_keypoints,), -1, dtype=torch.long, device=device)  # Track best version

#     # # Loop over descriptor versions
#     # for i in range(num_versions):
#     #     feats0 = {
#     #         'keypoints': pixel_base_keypoints[None].to(device),
#     #         'descriptors': affined_descriptions[i][None].to(device),
#     #         'image_size': base_img_size[None].to(device)
#     #     }
#     #     feats1 = {
#     #         'keypoints': pixel_detected_deformed_keypoints[None].to(device),
#     #         'descriptors': detected_deformed_descriptions[None].to(device),
#     #         'image_size': deformed_img_size[None].to(device)
#     #     }

#     #     affine_matches = lightglue_matcher({'image0': feats0, 'image1': feats1})

#     #     matches = affine_matches['matches'][0]  # Shape: (num_matches, 2)
#     #     scores = affine_matches['scores'][0]  # Shape: (num_matches,)

#     #     # Extract base and deformed keypoints
#     #     base_indices = matches[:, 0]  # Indices of matched keypoints in the base image
#     #     deformed_keypoints = feats1['keypoints'][0][matches[:, 1]]  # Matched keypoints
#     #     base_keypoints = feats0['keypoints'][0][matches[:, 0]]  # Corresponding base keypoints

#     #     # Use scatter_max to update the best match per keypoint efficiently
#     #     update_mask = scores > best_scores[base_indices]  # Mask for better scores
#     #     best_scores[base_indices] = torch.where(update_mask, scores, best_scores[base_indices])
#     #     best_matches[base_indices] = torch.where(update_mask[:, None], deformed_keypoints, best_matches[base_indices])
#     #     base_keypoint_carrier[base_indices] = torch.where(update_mask[:, None], base_keypoints, base_keypoint_carrier[base_indices])
#     #     best_descriptor_versions[base_indices] = torch.where(update_mask, i, best_descriptor_versions[base_indices])

#     # # Select top 200 unique matches based on score
#     # top_indices = best_scores.argsort(descending=True)[:200]

#     # affined_matches = base_keypoint_carrier[top_indices]  # Best 200 base keypoints
#     # deformed_matches = best_matches[top_indices]  # Best 200 matched deformed keypoints
#     # matched_transformations = best_descriptor_versions[top_indices]

#     stretch_counts = torch.bincount(matched_transformations.to(torch.int64))
#     index = 5 if len(stretch_counts) > 5 else len(stretch_counts)
#     top5_counts, top5_indices = torch.topk(stretch_counts, index)
#     top5_percents = (top5_counts / stretch_counts.sum()) * 100
#     print(f'Top 5 stretches for matching:')
#     for i in range(index):
#         print(f'{tensors[top5_indices[i]]}: {top5_percents[i]:.0f}%')


#         # Old version
#         # if len(best_scores) < 500:
#         #     best_base_matches.append(feats0['keypoints'][0][matches[:, 0]])
#         #     best_deformed_matches.append(feats1['keypoints'][0][matches[:, 1]])
#         #     best_scores = torch.cat([best_scores, scores])
#         # else:
#         #     # Combine current batch with existing best matches
#         #     combined_scores = torch.cat([best_scores, scores])
#         #     combined_base_matches = torch.cat(best_base_matches + [feats0['keypoints'][0][matches[:, 0]]])
#         #     combined_deformed_matches = torch.cat(best_deformed_matches + [feats1['keypoints'][0][matches[:, 1]]])

#         #     # Sort and keep top 500
#         #     top_indices = combined_scores.argsort(descending=True)[:500]
#         #     best_scores = combined_scores[top_indices]
#         #     best_base_matches = [combined_base_matches[top_indices]]
#         #     best_deformed_matches = [combined_deformed_matches[top_indices]]

#     # affined_matches = torch.cat(best_base_matches, dim=0)
#     # deformed_matches = torch.cat(best_deformed_matches, dim=0)

# gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in affined_matches.cpu()])
# distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
# good = (distances < good_match_threshold).sum().item()
# total = len(distances)
# accuracy = good / total
# print(f'Affined Accuracy: {accuracy} ({good} / {total})')

# affined_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, affined_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=False))
# if model == 'dedode':
#     affined_matches_image.save("Visualisations/matches_affined_dedode.png")
# elif model == 'superpoint':
#     affined_matches_image.save("Visualisations/matches_affined_sp.png")

#----------------------------------------------------------------------------------
#   Matching Stretched to Deformed
#----------------------------------------------------------------------------------

if affine_matching == 'dualsoftmax':
    stretched_matches, deformed_matches, batch_ids = stretcher_matcher.match(pixel_base_keypoints[None].to(device), stretched_descriptions.to(device),
            pixel_detected_deformed_keypoints[None].to(device), detected_deformed_descriptions.to(device),
            P_A = base_P, P_B = deformed_P,
            normalize = True, inv_temp=20, threshold = 0.03, stretch_type=stretch_type)
elif affine_matching == 'lightglue':
    # shape = affined_descriptions.shape
    # affined_base_descriptions = affined_descriptions.reshape(shape[0]*shape[1], shape[2])
    # print(affined_base_descriptions.shape)
    # affined_pixel_base_keypoints = pixel_base_keypoints.repeat(shape[0], 1)
    # print(affined_pixel_base_keypoints.shape)
    # feats0 = {'keypoints': affined_pixel_base_keypoints[None].to(device), 'descriptors': affined_base_descriptions[None].to(device), 'image_size': base_img_size[None].to(device)}
    # feats1 = {'keypoints': pixel_detected_deformed_keypoints[None].to(device), 'descriptors': detected_deformed_descriptions[None].to(device), 'image_size': deformed_img_size[None].to(device)}

    # affine_matches = lightglue_matcher({'image0': feats0, 'image1': feats1})

    # matches = affine_matches['matches'][0]
    # base_matches = feats0['keypoints'][0][matches[:,0]]
    # deformed_matches = feats1['keypoints'][0][matches[:,1]]

    # New Version
    best_matches = {}
    base_keypoint_carrier = {}
    best_scores = {}
    best_descriptor_versions = {}

    # Old Version
    # best_matches = []
    # best_base_matches = []
    # best_deformed_matches = []
    # best_scores = torch.tensor([], device=device)

    print(pixel_base_keypoints.shape)
    print(stretched_descriptions.shape)
    print(pixel_detected_deformed_keypoints.shape)
    print(detected_deformed_descriptions.shape)
    print(base_img_size.shape)
    print(deformed_img_size.shape)

    for i in range(stretched_descriptions.shape[0]):  # Iterate over batch

        feats0 = {
            'keypoints': pixel_base_keypoints[None].to(device),
            'descriptors': stretched_descriptions[i][None].to(device),
            'image_size': base_img_size[None].to(device)
        }
        feats1 = {
            'keypoints': pixel_detected_deformed_keypoints[None].to(device),
            'descriptors': detected_deformed_descriptions[None].to(device),
            'image_size': deformed_img_size[None].to(device)
        }

        affine_matches = lightglue_matcher({'image0': feats0, 'image1': feats1})

        matches = affine_matches['matches'][0]
        scores = affine_matches['scores'][0]  # Assuming LightGlue provides scores

        # New Version
        base_indices = matches[:, 0]  # Indices of matched keypoints in the base image
        base_keypoints = feats0['keypoints'][0][matches[:,0]]  # Corresponding matched keypoints
        deformed_keypoints = feats1['keypoints'][0][matches[:, 1]]  # Corresponding matched keypoints

        for j in range(len(base_indices)):
            keypoint_idx = base_indices[j].item()  # Convert to int for dictionary key
            score = scores[j].item()

            # If keypoint is not stored yet OR this match has a higher score, update it
            if keypoint_idx not in best_matches or score > best_scores[keypoint_idx]:
                best_matches[keypoint_idx] = deformed_keypoints[j]  # Save best-matching keypoint
                base_keypoint_carrier[keypoint_idx] = base_keypoints[j]  # Save corresponding base keypoint
                best_scores[keypoint_idx] = score  # Save best score
                best_descriptor_versions[keypoint_idx] = i  # Track descriptor version

    # Convert to tensors
    unique_base_matches = torch.stack(list(base_keypoint_carrier.values()))  # Base keypoints
    # unique_base_matches = torch.tensor(list(best_matches.keys()), device=device)  # Indices of best keypoints
    unique_deformed_matches = torch.stack(list(best_matches.values()))  # Matched keypoints
    unique_scores = torch.tensor(list(best_scores.values()), device=device)  # Best scores
    descriptor_versions = torch.tensor(list(best_descriptor_versions.values()), device=device)  # Descriptor versions

    # Select top 500 unique matches based on score
    top_indices = unique_scores.argsort(descending=True)[:200]

    stretched_matches = unique_base_matches[top_indices]  # Best 500 keypoints (indices)
    print(stretched_matches)
    print(stretched_matches.shape)
    deformed_matches = unique_deformed_matches[top_indices]  # Best 500 matched keypoints
    print(deformed_matches)
    print(deformed_matches.shape)
    matched_transformations = descriptor_versions[top_indices]  # Descriptor versions for these matches
    print(matched_transformations)
    print(matched_transformations.shape)

    stretch_counts = torch.bincount(matched_transformations.to(torch.int64))
    index = 5 if len(stretch_counts) > 5 else len(stretch_counts)
    top5_counts, top5_indices = torch.topk(stretch_counts, index)
    top5_percents = (top5_counts / stretch_counts.sum()) * 100
    print(f'Top 5 stretches for matching:')
    for i in range(index):
        print(f'{tensors[top5_indices[i]]}: {top5_percents[i]:.0f}%')


        # Old version
        # if len(best_scores) < 500:
        #     best_base_matches.append(feats0['keypoints'][0][matches[:, 0]])
        #     best_deformed_matches.append(feats1['keypoints'][0][matches[:, 1]])
        #     best_scores = torch.cat([best_scores, scores])
        # else:
        #     # Combine current batch with existing best matches
        #     combined_scores = torch.cat([best_scores, scores])
        #     combined_base_matches = torch.cat(best_base_matches + [feats0['keypoints'][0][matches[:, 0]]])
        #     combined_deformed_matches = torch.cat(best_deformed_matches + [feats1['keypoints'][0][matches[:, 1]]])

        #     # Sort and keep top 500
        #     top_indices = combined_scores.argsort(descending=True)[:500]
        #     best_scores = combined_scores[top_indices]
        #     best_base_matches = [combined_base_matches[top_indices]]
        #     best_deformed_matches = [combined_deformed_matches[top_indices]]

    # stretched_matches = torch.cat(best_base_matches, dim=0)
    # deformed_matches = torch.cat(best_deformed_matches, dim=0)

gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly) for pixel in stretched_matches.cpu()])
distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
good = (distances < good_match_threshold).sum().item()
total = len(distances)
accuracy = good / total
print(f'Affined Accuracy: {accuracy} ({good} / {total})')

stretched_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, stretched_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
if model == 'dedode':
    stretched_matches_image.save("Visualisations/matches_stretched_dedode.png")
elif model == 'superpoint':
    stretched_matches_image.save("Visualisations/matches_stretched_sp_dsm.png")

images = [
    baseline_keypoints_vis,
    deformed_keypoints_vis,
    detected_deformed_keypoints_vis,
    deformed_image_vis,
    affined_deformed_image_vis,
    stretched_deformed_image_vis,
    baseline_matches_image,
    # affined_matches_image,
    stretched_matches_image
    ]


images = [Image.fromarray(img[:,:,::-1]) if isinstance(img, np.ndarray) else img for img in images]

max_width = max(img.width for img in images)
max_height = max(img.height for img in images)

def pad_image(img, target_width, target_height):
    """Pads an image to the target size while keeping the original aspect ratio."""
    delta_w = target_width - img.width
    delta_h = target_height - img.height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(img, padding, (255, 255, 255))  # Pad with white background

padded_images = [pad_image(img, max_width, max_height) for img in images]

grid_width = max_width * 3
grid_height = max_height * 3

grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

positions = [
    (0, 0), (max_width, 0), (2 * max_width, 0),   # Top row (2 images)
    (0, max_height), (max_width, max_height), (2 * max_width, max_height),  # Middle row (3 images)
    (0, 2 * max_height), (max_width, 2 * max_height)#, (2 * max_width, 2 * max_height)  # Bottom row (3 images)
]

for i, img in enumerate(padded_images):
    grid_image.paste(img, positions[i])

# Save and show the combined image
grid_image.save("Visualisations/combined_grid_dsm.png")
grid_image.show()  # Show image for verification