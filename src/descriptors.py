"""SuperPoint keypoint detection and descriptor sampling.

The dense feature map SuperPoint produces can be interpolated at arbitrary
sub-pixel locations, which is what lets a keypoint detected in the rest image be
described again at its exact deformed position - the correspondence the training
pairs depend on (paper, Sec. 2.3.1).
"""

import torch
from lightglue import SuperPoint
from lightglue.utils import load_image
import matplotlib.pyplot as plt
import numpy as np
from .affine_transformations import apply_corotated_strain_with_keypoints

def custom_sample_descriptors(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor(
        [(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
    ).to(
        keypoints
    )[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {"align_corners": True} if torch.__version__ >= "1.3" else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", **args
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors

def plot_keypoints(image, keypoints):
    """Visualizes detected keypoints on an image."""
    plt.figure(figsize=(8, 6))
    plt.imshow(image.permute(1, 2, 0).cpu(), cmap="gray")
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c="r", s=5, label="Keypoints")
    plt.title("SuperPoint Keypoints")
    plt.legend()
    plt.axis("off")
    plt.show()

def sp_detect_and_describe(im, device, num_keypoints = 100):
    if im.dtype != np.uint8:
        im = np.array(im,dtype=np.uint8)
    if im.ndim == 3 and im.shape[-1] == 4:       # drop alpha; SuperPoint wants RGB
        im = im[:, :, :3]
    elif im.ndim == 2:                            # greyscale -> RGB
        im = np.repeat(im[:, :, None], 3, axis=2)
    extractor = SuperPoint().eval().to(device)
    im = torch.tensor(im.transpose((2,0,1)) / 255.0, dtype=torch.float).to(device)

    feats = extractor.extract(im, max_keypoints=num_keypoints)
    scales = feats["scales"].cpu()
    keypoints = feats["keypoints"][0].cpu()
    descriptions = feats["descriptors"][0].cpu()
    scores = feats["keypoint_scores"][0].cpu()
    dense_descriptions = feats["dense_descriptors"].cpu()
    img_size = feats["image_size"][0].cpu()

    return keypoints, scores, descriptions, dense_descriptions, scales, img_size



def sp_get_affine_deformed_descriptions(image, pixel_keypoints, tensors, device):
    """Ground-truth descriptors under each deformation, by re-running SuperPoint.

    This is the expensive test-time augmentation baseline of Sec. 2.1: it deforms
    the image and recomputes features once per hypothesis. Stretcher replaces it
    with a single latent-space pass. Kept as the reference used to generate
    training targets and to time the two approaches against each other.
    """
    affine_deformed_descriptions = []

    # Loop over deformations
    for j, deformation in enumerate(tensors):
        
        # Apply deformation
        deformed_image, deformed_pixel_keypoints = apply_corotated_strain_with_keypoints(image, pixel_keypoints, deformation, dataset_mode=False)

        # Get deformed descriptor without saving and reading the image
        deformed_image = np.array(deformed_image, dtype=np.uint8)
        _, _, _, deformed_dense_descriptions, scales, _ = sp_detect_and_describe(deformed_image, device, 10000)
        sp_keypoints = (torch.tensor(deformed_pixel_keypoints) + 0.5) * scales - 0.5
        deformed_descriptions = custom_sample_descriptors(sp_keypoints.to(float), deformed_dense_descriptions.cpu().to(float)).permute(0,2,1)[0]
        affine_deformed_descriptions.append(deformed_descriptions.cpu())
    
    return torch.stack(affine_deformed_descriptions)