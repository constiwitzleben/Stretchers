import torch
from lightglue import SuperPoint
from lightglue.utils import load_image
import matplotlib.pyplot as plt
import numpy as np
from .affine_transformations import apply_corotated_strain_with_keypoints

# ------------------------------------------------------------------------------
# Inline extraction of a descriptor at a specific (x, y) location (e.g., (100, 200))
# ------------------------------------------------------------------------------

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

# ----------------------
# OPTIONAL: Visualization of keypoints
# ----------------------
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
    h, w = im.shape[:2]
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



if __name__ == "__main__":

    # Disable gradients for inference
    torch.set_grad_enabled(False)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load SuperPoint extractor (without limiting keypoints)
    extractor = SuperPoint().eval().to(device)  # No max_num_keypoints, default is None

    # Load test image
    image0 = load_image("data/medical_deformed/brain.png")  # Returns a torch tensor
    image0 = image0.to(device)

    # Extract SuperPoint features
    feats0 = extractor.extract(image0)

    # Retrieve keypoints, descriptors, and scores
    keypoints0 = feats0["keypoints"][0].cpu()  # (N, 2) keypoints in (x, y)
    descriptors0 = feats0["descriptors"][0].cpu()  # (N, 256) descriptors
    scores0 = feats0["keypoint_scores"][0].cpu()  # (N,) confidence scores

    print(f"Extracted {keypoints0.shape[0]} keypoints.")

    # Define the desired (x, y) coordinate
    x, y = 100, 200

    # Create a keypoint tensor with the desired coordinate.
    # It must be a float tensor on the same device.
    # Expected shape for sample_descriptors is (batch, num_points, 2), so we add the necessary dimensions.
    keypoint_tensor = torch.tensor([[x, y]], dtype=torch.float32, device=device)  # Shape: (1, 2)
    keypoint_tensor = keypoint_tensor.unsqueeze(0)  # Now shape: (1, 1, 2)

    # Use the extractor's sample_descriptors method with a scale factor of 8 (typical for SuperPoint)
    # The output has shape (B, descriptor_dim, num_points); here we index batch 0 and the first (and only) keypoint.
    descriptor_at_xy = custom_sample_descriptors(keypoint_tensor, descriptors0, 8)[0, :, 0]
    print(f"Descriptor at location ({x}, {y}) has shape: {descriptor_at_xy.shape}")

    # ----------------------
    # OPTIONAL: Visualization of keypoints
    # ----------------------
    def plot_keypoints(image, keypoints):
        """Visualizes detected keypoints on an image."""
        plt.figure(figsize=(8, 6))
        plt.imshow(image.permute(1, 2, 0).cpu(), cmap="gray")
        plt.scatter(keypoints[:, 0], keypoints[:, 1], c="r", s=5, label="Keypoints")
        plt.title("SuperPoint Keypoints")
        plt.legend()
        plt.axis("off")
        plt.show()

    plot_keypoints(image0, keypoints0)

def sp_get_affine_deformed_descriptions(image, pixel_keypoints, tensors, device):

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