import os
import time
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher

from .matching_util import draw_matches, draw_matching_comparison
from .models import TripleNet
from .affine_transformations import generate_strain_tensors, apply_corotated_strain_with_keypoints
from .dsm_matching import StretcherDualSoftMaxMatcher
from .descriptors import sp_detect_and_describe, custom_sample_descriptors

import cv2

# The FEniCS/PyVista import (and the pkg-config repair it needs) lives in
# fenics_deformation, which owns the FEM stack for the package.
from .fenics_deformation import (
    fe, pv, _FEM_AVAILABLE, _require_fem,
    create_deformed_medical_image_pair, track_pixel_displacement,
    bottom, epsilon, sigma, von_mises_stress, von_mises_strain, get_strain,
)

def get_best_device(verbose = False):
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if verbose: print (f"Fastest device found is: {device}")
    return device

def load_images(im_path, deformed_im_path, figsize=(10, 5), titles=("Base image", "Deformed image")):
    """
    Load two images, strip alpha channels if present, display them side by side,
    and return numpy arrays (RGB) for downstream processing.
    """
    image = Image.open(im_path)
    deformed_image = Image.open(deformed_im_path)

    image_np = np.array(image)
    if image_np.ndim == 3 and image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]

    deformed_np = np.array(deformed_image)
    if deformed_np.ndim == 3 and deformed_np.shape[-1] == 4:
        deformed_np = deformed_np[:, :, :3]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(image_np)
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    axes[1].imshow(deformed_np)
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()

    return image_np, deformed_np



def extract_superpoint_keypoints(
    im_path,
    deformed_im_path,
    device,
    num_keypoints=512,
):
    """
    Extract SuperPoint keypoints/descriptors for two images and return
    features plus convenience tensors used later in matching.

    Returns (in order):
    - feats0, feats1: dictionaries from SuperPoint.extract
    - base_keypoints, base_descriptors, base_scores
    - deformed_keypoints, deformed_descriptors, deformed_scores
    - extractor, dsm_matcher (only if return_models=True)
    """
    extractor = SuperPoint(max_num_keypoints=num_keypoints).eval().to(device)

    start = time.time()
    image0 = load_image(im_path).to(device)
    image1 = load_image(deformed_im_path).to(device)
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    end = time.time()
    print(f"Description time: {end - start:.3f}s")

    return (
        feats0,
        feats1
    )


def matching(
    matcher,
    feats0,
    feats1,
    image_np,
    deformed_np,
    device,
    inv_temp=20,
    dsm_threshold=0.01,
    save_dir='Visualisations',
    save_name='matches_baseline.png',
    image = True,
):
    """
    Step 4 baseline matching controlled by `matcher` ('dsm' or 'lightglue').

    Returns:
      baseline_img (PIL.Image.Image), base_matches (torch.Tensor), def_matches (torch.Tensor)
    """
    os.makedirs(save_dir, exist_ok=True)

    if matcher == 'dsm':
        dsm_matcher = DualSoftMaxMatcher()

        base_keypoints = feats0['keypoints']
        base_descriptors = feats0['descriptors']
        base_scores = feats0['keypoint_scores'][0]

        deformed_keypoints = feats1['keypoints']
        deformed_descriptors = feats1['descriptors']
        deformed_scores = feats1['keypoint_scores'][0]

        start = time.time()
        base_matches, def_matches, _ = dsm_matcher.match(
            base_keypoints.to(device), base_descriptors.to(device),
            deformed_keypoints.to(device), deformed_descriptors.to(device),
            P_A=base_scores, P_B=deformed_scores,
            normalize=True, inv_temp=inv_temp, threshold=dsm_threshold,
        )
        end = time.time()
        print(f"Baseline DSM matching time: {end - start:.3f}s")

    elif matcher == 'lightglue':
        lg = LightGlue(features='superpoint').eval().to(device)
        start = time.time()
        out = lg({'image0': feats0, 'image1': feats1})
        matches = out['matches'][0]
        base_matches = feats0['keypoints'][0][matches[:, 0]]
        def_matches = feats1['keypoints'][0][matches[:, 1]]
        end = time.time()
        print(f"Baseline LightGlue matching time: {end - start:.3f}s")

    else:
        print('Matcher not implemented')

    baseline_img = None
    if image:
        from PIL import Image  # local import to avoid circulars
        baseline_img = Image.fromarray(
            draw_matches(image_np, base_matches.cpu(), deformed_np, def_matches.cpu())
        )

        # Show and save
        plt.figure(figsize=(10, 5))
        plt.imshow(baseline_img)
        plt.title('Baseline matches')
        plt.axis('off')
        plt.show()

        baseline_path = os.path.join(save_dir, save_name)
        baseline_img.save(baseline_path)

    return baseline_img, base_matches, def_matches


def stretch_descriptions(
    features,
    device,
    model_path,
    hidden_dim=2048,
    num_layers=2,
):
    """
    Step 5: load the stretcher and produce stretched descriptors for feats0.

    Inputs:
      feats0: SuperPoint features dict for image0
      device: torch device
      model_path: path to stretcher weights
      hidden_dim, num_layers: TripleNet config
      stretch_type: 'normal' or 'only27' (controls strain tensors)

    Returns:
      stretched (torch.Tensor on device) with shape [num_strains, N, D]
    """
    tensors = generate_strain_tensors()

    stretcher = TripleNet(256, 3, hidden_dim=hidden_dim, num_layers=num_layers).float().to(device)
    stretcher.load_state_dict(torch.load(model_path, map_location=device))
    stretcher.eval()

    start = time.time()
    base_descriptions = features['descriptors'][0].cpu()
    with torch.no_grad():
        stretched_descriptions = np.array([stretcher(base_descriptions.to(torch.float32).to(device), torch.tensor(tensor).to(torch.float32).to(device).repeat(len(base_descriptions),1)).cpu() for tensor in tensors])
    stretched_descriptions = torch.tensor(stretched_descriptions).to(device)
    end = time.time()
    print(f"Stretching time: {end - start:.3f}s")

    return stretched_descriptions


def stretched_matching(
    matcher,
    feats0,
    feats1,
    stretched_descriptions,
    image0,
    image1,
    device,
    inv_temp=20,      # paper setting; see scripts/evaluate_table1.py
    dsm_threshold=0.03,
    topk=500,
    baseline_img=None,
    save_dir='Visualisations',
    save_name='stretched_matches.png',
    comparison_name='matches_comparison.png',
    image = True,
):
    """
    Step 6: run matching using stretched descriptors and optionally save a comparison.

    Returns:
      stretched_img (PIL.Image.Image), stretched_matches (torch.Tensor), def_matches_st (torch.Tensor)
    """
    os.makedirs(save_dir, exist_ok=True)

    from PIL import Image  # local import

    if matcher == 'dsm':
        base_keypoints = feats0['keypoints']
        deformed_keypoints = feats1['keypoints']
        deformed_descriptors = feats1['descriptors']
        base_scores = feats0['keypoint_scores'][0]
        deformed_scores = feats1['keypoint_scores'][0]

        matcher_obj = StretcherDualSoftMaxMatcher()
        start = time.time()
        stretched_matches, def_matches_st, _ = matcher_obj.match(
            base_keypoints.to(device), stretched_descriptions.to(device),
            deformed_keypoints.to(device), deformed_descriptors.to(device),
            P_A=base_scores, P_B=deformed_scores,
            normalize=True, inv_temp=inv_temp, threshold=dsm_threshold,
        )
        end = time.time()
        print(f"Stretched DSM matching time: {end - start:.3f}s")

    else:
        lg = LightGlue(features='superpoint').eval().to(device)

        start = time.time()
        best_matches = {}
        best_scores = {}
        base_carrier = {}

        num_strains = stretched_descriptions.shape[0]
        for i in range(num_strains):
            feats0['descriptors'] = stretched_descriptions[i][None].to(device)
            out = lg({'image0': feats0, 'image1': feats1})
            matches = out['matches'][0]
            scores = out['scores'][0]

            base_indices = matches[:, 0]
            base_kps = feats0['keypoints'][0][matches[:, 0]]
            def_kps = feats1['keypoints'][0][matches[:, 1]]

            for j in range(len(base_indices)):
                idx = int(base_indices[j].item())
                score = float(scores[j].item())
                if idx not in best_matches or score > best_scores[idx]:
                    best_matches[idx] = def_kps[j]
                    base_carrier[idx] = base_kps[j]
                    best_scores[idx] = score

        if len(best_matches) == 0:
            raise RuntimeError("No LightGlue matches found across stretched descriptors.")

        import torch
        unique_base = torch.stack(list(base_carrier.values()))
        unique_def = torch.stack(list(best_matches.values()))
        unique_scores = torch.tensor(list(best_scores.values()), device=device)

        k = min(topk, unique_scores.numel())
        top_idx = unique_scores.argsort(descending=True)[:k]
        stretched_matches = unique_base[top_idx]
        def_matches_st = unique_def[top_idx]

        end = time.time()
        print(f"Stretched LightGlue matching time: {end - start:.3f}s")


    stretched_img = None
    if image:
        stretched_img = Image.fromarray(
            draw_matches(image0, stretched_matches.cpu(), image1, def_matches_st.cpu())
        )

        # Visualize and save comparison if provided
        plt.figure(figsize=(10, 5))
        plt.imshow(stretched_img)
        plt.title('Stretched matches')
        plt.axis('off')
        plt.show()

        stretched_path = os.path.join(save_dir, save_name)
        stretched_img.save(stretched_path)

        if baseline_img is not None:
            comparison_path = os.path.join(save_dir, comparison_name)
            draw_matching_comparison(baseline_img, stretched_img, comparison_path)
            print(f"Saved comparison to: {comparison_path}")

    return stretched_img, stretched_matches, def_matches_st


def save_dataset(
    non_deformed_descriptors,
    deformed_descriptors,
    deformation_idx,
    output_path='data/SuperPoint_Descriptors_Dataset_Test.pth',
):
    """
    Convert numpy arrays to torch tensors and save to a single .pth file.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    torch_descriptors = torch.tensor(non_deformed_descriptors)
    torch_deformed_descriptors = torch.tensor(deformed_descriptors)
    torch_deformations = torch.tensor(deformation_idx)
    torch.save(
        {
            'descriptors': torch_descriptors,
            'deformed_descriptors': torch_deformed_descriptors,
            'transformations': torch_deformations,
        },
        output_path,
    )
    print(f'Saved dataset to: {output_path}')


def create_dataset(
    image_dir,
    device,
    deformations_per_image,
    kp_per_deformation,
    max_images=None
):
    """
    Encapsulate the Step 2 image loop for building SuperPoint descriptor arrays.

    Returns:
      non_deformed_descriptors (np.ndarray), deformed_descriptors (np.ndarray)
    """

    kp_per_image = deformations_per_image*kp_per_deformation

    image_names = os.listdir(image_dir)
    num_images = len(image_names) if max_images is None else max_images
    print(f'Number of images: {num_images}')
    non_deformed_descriptors = np.zeros((num_images*deformations_per_image*kp_per_deformation, 256))
    deformed_descriptors = np.zeros((num_images*deformations_per_image*kp_per_deformation, 256))

    # Prepare list of deformations
    deformation_grid = generate_strain_tensors()
    deformation_idx = np.tile(np.arange(len(deformation_grid)), (num_images, 1))

    for i, image_name in enumerate(image_names[:max_images]):
        image_path = os.path.join(image_dir, image_name)
        image = Image.open(image_path)
        W, H = image.size
        image = image.resize((W // 2, H // 2))
        image = np.array(image, dtype=np.uint8)

        keypoints, scores, descriptions, dense_descriptions, scales, img_size = sp_detect_and_describe(image, device, kp_per_image)
        top_indices = scores.argsort(descending=True)[:kp_per_image]
        keypoints = keypoints[top_indices]
        descriptions = descriptions[top_indices]

        non_deformed_descriptors[i*kp_per_deformation*deformations_per_image:(i+1)*kp_per_deformation*deformations_per_image] = descriptions.cpu()

        deformations = np.array(deformation_grid)[deformation_idx[i, :]]
        keypoints = keypoints.reshape(deformations_per_image, kp_per_deformation, 2)

        for j, (deformation, keypoint_set) in enumerate(zip(deformations, keypoints)):
            deformed_image, deformed_keypoint_set = apply_corotated_strain_with_keypoints(image, keypoint_set, deformation, dataset_mode=False)

            _, _, _, deformed_dense_descriptions, scales, _ = sp_detect_and_describe(deformed_image, device, 10000)
            sp_keypoints = (torch.tensor(deformed_keypoint_set[0]) + 0.5) * scales - 0.5
            deformed_description_set = custom_sample_descriptors(sp_keypoints.to(torch.float32), deformed_dense_descriptions.cpu()).permute(0, 2, 1)[0]

            deformed_descriptors[
                i*kp_per_deformation*deformations_per_image + j*kp_per_deformation:
                i*kp_per_deformation*deformations_per_image + (j+1)*kp_per_deformation
            ] = deformed_description_set.cpu()

        if i % 100 == 0:
            print(f"Processed {i} images")

    deformation_idx = deformation_idx.flatten()
    parameters = torch.tensor(np.array(deformation_grid)[deformation_idx])

    return non_deformed_descriptors, deformed_descriptors, parameters

# The FEM synthetic-deformation pipeline lives in fenics_deformation, which is
# the single implementation used by both the notebooks and
# scripts/evaluate_table1.py. These re-exports keep the notebook imports stable.

def evaluate_matches(base_matches, deformed_matches, deformation_info):
    gt_pixel_coords = np.array([track_pixel_displacement(pixel, deformation_info) for pixel in base_matches.cpu()])
    distances = (deformed_matches.cpu() - gt_pixel_coords).norm(dim=1)
    good = (distances < 5).sum().item()
    total = len(distances)
    accuracy = good / total
    print(f'Baseline Accuracy: {accuracy} ({good} / {total})')
    return distances