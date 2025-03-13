from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from DeDoDe.utils import *
from PIL import Image
import numpy as np
from lightglue import LightGlue, DISK, ALIKED, SuperPoint
from lightglue.utils import load_image, rbd
from fenics_2D_elasticity_grid_proj import create_deformed_medical_image_pair, track_pixel_displacement, get_strain
from util.matching import draw_matches_with_scores, strain_entropy, strain_balanced_precision
from matchers.max_similarity import StretcherDualSoftMaxMatcher
from models import TripleNet
from util.Affine_Transformations import generate_strain_tensors
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message="__array_wrap__ must accept context and return_scalar")

good_match_threshold = 5
model_dir = "models/spstretcher_new.pth"
# pretty_im_path = 'data/pretty_data/rest.png'
# pretty_deformed_im_path = 'data/pretty_data/deformed.png'
pretty_im_path = 'data/medical_deformed/pig_liver_to_elongate.png'
pretty_deformed_im_path = 'data/medical_deformed/pig_liver_to_elongate_deformed.png'
im_path = 'data/medical_deformed/pig_liver_to_elongate.png'
deformed_im_path = 'data/medical_deformed/pig_liver_to_elongate_deformed.png'
deformations = np.array([[8e6, 1e6], [8e6, -1e6], [-4e6, 2e6], [-4e6, -2e6]])

device = get_best_device()

hidden_dim = 2048
num_layers = 2
stretcher = TripleNet(256,3,hidden_dim=hidden_dim,num_layers = num_layers).float().to(device)
stretcher.load_state_dict(torch.load(model_dir,map_location=device))
stretcher.eval()

stretcher_dsm_dict = {'num_matches': [], 'precision': [], 'matching_score': [], 'entropy': [], 'sb_precision': []}
stretcher_lg_dict = {'num_matches': [], 'precision': [], 'matching_score': [], 'entropy': [], 'sb_precision': []}

for i, deformation in enumerate(deformations):
    i += 1

    u, s, new_lx, new_ly, bottom_left = create_deformed_medical_image_pair(im_path, deformed_im_path, deformation[0], deformation[1])

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

    #---------------------------------------------------------------------------------------------------------------------------------

    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    lg_matcher = LightGlue(features='superpoint').eval().to(device)
    dsm_matcher = StretcherDualSoftMaxMatcher()

    image0 = load_image(im_path).to(device)
    image1 = load_image(deformed_im_path).to(device)
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    #--------------------------------------------------------------------------------------------------------------------------

    base_keypoints = feats0['keypoints']
    base_descriptions = feats0['descriptors']
    base_P = feats0['keypoint_scores'][0]
    deformed_keypoints = feats1['keypoints']
    deformed_descriptions = feats1['descriptors']
    deformed_P = feats1['keypoint_scores'][0]

    tensors = np.array(generate_strain_tensors())
    with torch.no_grad():
        stretched_descriptions = np.array([stretcher(base_descriptions[0].cpu().to(torch.float32).to(device), torch.tensor(tensor).to(torch.float32).to(device).repeat(len(base_descriptions[0]),1)).cpu() for tensor in tensors])
    stretched_descriptions = torch.tensor(stretched_descriptions).to(device)

    base_matches, deformed_matches, batch_ids = dsm_matcher.match(base_keypoints.to(device), stretched_descriptions.to(device),
                deformed_keypoints.to(device), deformed_descriptions.to(device),
                P_A = base_P, P_B = deformed_P,
                normalize = True, inv_temp=20, threshold = 0.03)

    gt_pixel_coords = np.array([track_pixel_displacement(u, pixel, W, H, dW, dH, 10, 10, new_lx, new_ly, bottom_left) for pixel in base_matches.cpu()])
    distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
    good_mask = distances < good_match_threshold
    num_good_matches = good_mask.sum().item()

    num_matches = len(distances)
    precision = num_good_matches / num_matches
    matching_score = num_good_matches / min(base_keypoints.shape[1], deformed_keypoints.shape[1])
    entropy = strain_entropy(s, base_matches[good_mask].cpu(), W, H)
    sb_precision = strain_balanced_precision(s, base_matches.cpu(), good_mask, W, H)

    stretcher_dsm_dict['num_matches'].append(num_matches)
    stretcher_dsm_dict['precision'].append(precision)
    stretcher_dsm_dict['matching_score'].append(matching_score)
    stretcher_dsm_dict['entropy'].append(entropy)
    stretcher_dsm_dict['sb_precision'].append(sb_precision)

    print(f'Stretcher DSM Precision: {precision} ({num_good_matches} / {num_matches})')
    print(f'Stretcher DSM Matching Score: {matching_score}')
    print(f'Stretcher DSM Number of Matches: {num_matches}')
    print(f'Stretcher DSM Entropy: {entropy}')
    print(f'Stretcher DSM Strain Balanced Precision: {sb_precision}')

    baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
    baseline_matches_image.save(f'Visualisations/{i}/matches_Stretcher_dsm.png')

    #--------------------------------------------------------------------------------------------------------------------------

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

    stretcher_lg_dict['num_matches'].append(num_matches)
    stretcher_lg_dict['precision'].append(precision)
    stretcher_lg_dict['matching_score'].append(matching_score)
    stretcher_lg_dict['entropy'].append(entropy)
    stretcher_lg_dict['sb_precision'].append(sb_precision)

    print(f'Stretcher LG Precision: {precision} ({num_good_matches} / {num_matches})')
    print(f'Stretcher LG Matching Score: {matching_score}')
    print(f'Stretcher LG Number of Matches: {num_matches}')
    print(f'Stretcher LG Entropy: {entropy}')
    print(f'Stretcher LG Strain Balanced Precision: {sb_precision}')

    baseline_matches_image = Image.fromarray(draw_matches_with_scores(pretty_image, base_matches.cpu(), pretty_deformed_image, deformed_matches.cpu(), distances, good_match_threshold, lines=True))
    baseline_matches_image.save(f'Visualisations/{i}/matches_Stretcher_lg.png')

    #--------------------------------------------------------------------------------------------------------------------------

# Calculate mean and std of all scores and print them
stretcher_dsm_dict = {key: np.array(val) for key, val in stretcher_dsm_dict.items()}
stretcher_lg_dict = {key: np.array(val) for key, val in stretcher_lg_dict.items()}

stretcher_dsm_mean = {key: np.mean(val) for key, val in stretcher_dsm_dict.items()}
stretcher_lg_mean = {key: np.mean(val) for key, val in stretcher_lg_dict.items()}

stretcher_dsm_std = {key: np.std(val) for key, val in stretcher_dsm_dict.items()}
stretcher_lg_std = {key: np.std(val) for key, val in stretcher_lg_dict.items()}

print('Stretcher DSM Results:')
print(f'Number of Matches: {stretcher_dsm_mean["num_matches"]} +/- {stretcher_dsm_std["num_matches"]}')
print(f'Precision: {stretcher_dsm_mean["precision"]} +/- {stretcher_dsm_std["precision"]}')
print(f'Matching Score: {stretcher_dsm_mean["matching_score"]} +/- {stretcher_dsm_std["matching_score"]}')
print(f'Entropy: {stretcher_dsm_mean["entropy"]} +/- {stretcher_dsm_std["entropy"]}')
print(f'Strain Balanced Precision: {stretcher_dsm_mean["sb_precision"]} +/- {stretcher_dsm_std["sb_precision"]}')

print('Stretcher LG Results:')
print(f'Number of Matches: {stretcher_lg_mean["num_matches"]} +/- {stretcher_lg_std["num_matches"]}')
print(f'Precision: {stretcher_lg_mean["precision"]} +/- {stretcher_lg_std["precision"]}')
print(f'Matching Score: {stretcher_lg_mean["matching_score"]} +/- {stretcher_lg_std["matching_score"]}')
print(f'Entropy: {stretcher_lg_mean["entropy"]} +/- {stretcher_lg_std["entropy"]}')
print(f'Strain Balanced Precision: {stretcher_lg_mean["sb_precision"]} +/- {stretcher_lg_std["sb_precision"]}')