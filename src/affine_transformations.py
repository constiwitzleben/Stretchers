import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp, AffineTransform
from numpy import pad
from skimage import io
import torch
import cv2

# Generate a simple chessboard image
def create_chessboard(size=100, block_size=10):
    image = np.zeros((size, size))
    for i in range(size // block_size):
        for j in range(size // block_size):
            if (i + j) % 2 == 0:
                image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = 1
    return image


 # Generate 125 strain tensors
def generate_strain_tensors():
    strain_xx = np.array([-0.5, -0.25, 0.0, 0.5, 1.0])  # Stretching values
    strain_yy = np.array([-0.5, -0.25, 0.0, 0.5, 1.0])  # Stretching values
    # shear_xy1 = np.linspace(-0.9, -0.5, 2)   # Shear strain
    # shear_xy2 = np.linspace(0.5, 0.9, 2)   # Shear strain
    shear_xy = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    tensors = [
        (xx, yy, xy)
        for xx in strain_xx
        for yy in strain_yy
        for xy in shear_xy
    ]
    return tensors

def generate_larger_strain_tensors():
    strain_xx = np.array([-0.66, -0.33, 0.0, 0.75, 1.5])  # Stretching values
    strain_yy = np.array([-0.66, -0.33, 0.0, 0.75, 1.5])  # Stretching values
    # shear_xy1 = np.linspace(-0.9, -0.5, 2)   # Shear strain
    # shear_xy2 = np.linspace(0.5, 0.9, 2)   # Shear strain
    shear_xy = np.array([-0.4, -0.2, 0.0, 0.2, 0.4])
    tensors = [
        (xx, yy, xy)
        for xx in strain_xx
        for yy in strain_yy
        for xy in shear_xy
    ]
    return tensors

def generate_27_strain_tensors():
    strain_xx = np.array([-0.5, 0.0, 1.0])  # Stretching values
    strain_yy = np.array([-0.5, 0.0, 1.0])  # Stretching values
    # shear_xy1 = np.linspace(-0.9, -0.5, 2)   # Shear strain
    # shear_xy2 = np.linspace(0.5, 0.9, 2)   # Shear strain
    shear_xy = np.array([-0.4, 0.0, 0.4])
    tensors = [
        (xx, yy, xy)
        for xx in strain_xx
        for yy in strain_yy
        for xy in shear_xy
    ]
    return tensors


def generate_test_strain_tensors():
    return [[0.5,0.5,0.0],[0.0,0.0,0.0],[-0.25,-0.25,0.0]]

# Perform polar decomposition
def polar_decomposition(F):
    U, S, Vt = np.linalg.svd(F)
    R = U @ Vt  # Rotation matrix
    U_stretch = Vt.T @ np.diag(S) @ Vt  # Pure strain component
    return R, U_stretch

# Create deformation gradient from strain tensor
def deformation_gradient_from_strain(strain_tensor):
    F = np.eye(2) + strain_tensor
    return F

# Apply transformation
def apply_corotated_strain(image, s):

    strain_tensor = np.array([[s[0], s[2]], 
                              [s[2], s[1]]])
    F = deformation_gradient_from_strain(strain_tensor)

    R, F_strain = polar_decomposition(F)

    # Build affine transform matrix (add homogeneous coordinates)
    F_strain_h = np.eye(3)
    F_strain_h[:2, :2] = F_strain

    # Center image for proper transformation
    center = np.array(image.shape[:2]) / 2
    translation_to_origin = np.eye(3)
    translation_to_origin[:2, 2] = -center

    translation_back = np.eye(3)
    translation_back[:2, 2] = center

    # Combine transformations
    affine_matrix = translation_back @ F_strain_h @ translation_to_origin
    transform = AffineTransform(matrix=affine_matrix)

    # Apply warp
    transformed_image = warp(
        image,
        inverse_map=transform.inverse,
        mode="constant",
        cval=0.0,
        preserve_range=True,
        order=3  # Bicubic interpolation
    )
    return transformed_image


# Apply the strain tensor and transformation
def apply_strain(image, s, keypoints=None):
    """
    Apply a strain tensor to deform the 2D image.
    Parameters:
    - image: 2D numpy array, input image
    - strain_tensor: 2x2 numpy array, strain tensor
    - padding: int, padding to add around the image
    Returns:
    - deformed_image: 2D numpy array, transformed image
    """
    
    strain_tensor = np.array([[s[0], s[2]], 
                              [s[2], s[1]]])

    # Transformation matrix: F = I + ε
    transformation_matrix = np.eye(2) + strain_tensor

    def transform(coords):
        """
        Map the output coordinates to the input coordinates using the strain transformation.
        """
        coords = np.dot(coords, np.linalg.inv(transformation_matrix).T)
        return coords

    # Use skimage.transform.warp with higher interpolation order to minimize aliasing
    deformed_image = warp(
        image, 
        transform, 
        mode='constant', 
        cval=0.0, 
        preserve_range=True, 
        order=3  # Bicubic interpolation
    )

    # Apply the same transformation to the keypoints
    if keypoints is not None:
        keypoints = transform(keypoints)

    return deformed_image, keypoints


# Display images in pages of 25
def display_images_in_pages(image, tensors, rows=5, cols=5):
    total_tensors = len(tensors)
    tensors_per_page = rows * cols
    pages = total_tensors // tensors_per_page + (1 if total_tensors % tensors_per_page else 0)

    for page in range(pages):
        print(f"Displaying page {page + 1} of {pages}")
        start_idx = page * tensors_per_page
        end_idx = start_idx + tensors_per_page
        tensors_on_page = tensors[start_idx:end_idx]

        fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
        axes = axes.ravel()

        for i, tensor in enumerate(tensors_on_page):
            deformed_image = apply_corotated_strain(image, tensor)

            height, width = deformed_image.shape[:2]
            center_y, center_x = height // 2, width // 2

            crop_size = 64
            top_left_y = center_y - crop_size // 2
            top_left_x = center_x - crop_size // 2

            # Crop the image
            cropped_patch = deformed_image[top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]

            axes[i].imshow(deformed_image, cmap='gray', vmin=0, vmax=1)
            #axes[i].imshow(cropped_patch, cmap='gray')
            axes[i].axis('off')
            axes[i].set_title(f"{tensor[0]:.2f},{tensor[1]:.2f},{tensor[2]:.2f}")

        # Remove unused subplots
        for ax in axes[len(tensors_on_page):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        # Pause to allow navigation between pages
        input(f"Press Enter to see the next page (page {page + 2} of {pages})...")


def apply_transformation_from_index(image, index, origin):
    tensors = generate_strain_tensors()
    



# Main
# image_size = 100
# block_size = 5
# padding = 100  # Prevent clipping

# # Create and pad chessboard image
# image = create_chessboard(size=image_size, block_size=block_size)
# image = io.imread('patch.png', as_gray=True)
# image = pad(image, padding, mode="constant", constant_values=0)

'''
# Define a strain tensor
strain_tensor = np.array([[0.12, 0.5], 
                          [0.5, 0.12]])  # Example high-strain tensor

# Compute deformation gradient
F = deformation_gradient_from_strain(strain_tensor)

# Perform polar decomposition
R, F_strain = polar_decomposition(F)

# Apply corotated strain transformation
corotated_deformed_image = apply_corotated_strain(image, F_strain)

# Apply pure strain transformation
deformed_image = apply_strain(image, F_strain)




# Plot the results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

#plt.subplot(1, 3, 2)
#plt.title("Deformed Image ")
#plt.imshow(deformed_image, cmap="gray")
#plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Deformed Image (Co-Rotational)")
plt.imshow(corotated_deformed_image, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
'''

# # Generate strain tensors
# strain_tensors = generate_strain_tensors()
# print(strain_tensors)
    
# # Display images in pages of 25
# display_images_in_pages(image, strain_tensors)

def apply_corotated_strain_with_keypoints(image, keypoints, s, dataset_mode=True):

    H,W = image.shape[:2]
    output_shape = (H, W)

    deformation = np.array(s)

    # Check if padding is needed
    padding = None

    if dataset_mode:
        if deformation[2] >= 0.4 or deformation[2] <= -0.4:
            if deformation[0] + deformation[1] <= -1.0:
                padding = 1
                print('padding with 1')
            elif deformation[0] + deformation[1] <= -0.75:
                padding = 0.2
                print('padding with 0.2')
    

    if padding is not None:
        hpadding = int(padding*H)
        wpadding = int(padding*W)
        image = np.pad(image, ((hpadding, hpadding), (wpadding, wpadding), (0, 0)), mode='reflect')

        keypoints = keypoints + torch.tensor([wpadding, hpadding])

    strain_tensor = np.array([[s[0], s[2]], 
                              [s[2], s[1]]])
    F = deformation_gradient_from_strain(strain_tensor)

    R, F_strain = polar_decomposition(F)

    # print(f'F_strain: {F_strain}')

    # Build affine transform matrix (add homogeneous coordinates)
    F_strain_h = np.eye(3)
    F_strain_h[:2, :2] = F_strain

    # Center image for proper transformation
    H,W = image.shape[:2]
    center = np.array([W,H]) / 2
    translation_to_origin = np.eye(3)
    translation_to_origin[:2, 2] = -center

    translation_back = np.eye(3)
    translation_back[:2, 2] = center

    # Combine transformations
    affine_matrix = translation_back @ F_strain_h @ translation_to_origin
    transform = AffineTransform(matrix=affine_matrix)

    if dataset_mode == False:
        corners = np.array([
        [0, 0],  # Top-left
        [W, 0],  # Top-right
        [W, H],  # Bottom-right
        [0, H]   # Bottom-left
        ])
        new_corners = transform(corners)
        min_x, min_y = np.floor(new_corners.min(axis=0)).astype(int)
        max_x, max_y = np.ceil(new_corners.max(axis=0)).astype(int)

        # Compute new width and height
        output_shape = (max_y - min_y, max_x - min_x)

        offset_transform = AffineTransform(translation=(-min_x, -min_y))
        transform = transform + offset_transform  # Combine transformations


    # Apply warp with skimage warp
    # transformed_image = warp(
    #     image,
    #     inverse_map=transform.inverse,
    #     mode="constant",
    #     cval=0.0,
    #     preserve_range=True,
    #     order=1, # Bicubic interpolation
    #     output_shape=output_shape
    # )

    # Apply warp with cv2 warpAffine
    M = transform.params[:2, :]
    transformed_image = cv2.warpAffine(image, M, (output_shape[1],output_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    transformed_image = np.array(transformed_image, dtype=np.uint8)

    # Apply the same transformation to the keypoints
    # Flip the coordinates of the keypoints

    # keypoints_h = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
    # transformed_keypoints = keypoints_h @ affine_matrix.T
    transformed_keypoints = transform(keypoints)
    # transformed_keypoints = transformed_keypoints[:, :2] / transformed_keypoints[:, 2:3]

    if padding is not None:
        transformed_keypoints = transformed_keypoints - np.array([wpadding, hpadding])
        transformed_image = transformed_image[hpadding:-hpadding, wpadding:-wpadding, :]

    return transformed_image, transformed_keypoints[None, :, :]

def transform_keypoints(image, keypoints, s):

    H,W = image.shape[:2]

    deformation = np.array(s)

    # Check if padding is needed
    padding = None
    if deformation[2] >= 0.4 or deformation[2] <= -0.4:
        if deformation[0] + deformation[1] <= -1.0:
            padding = 1
            print('padding with 1')
        elif deformation[0] + deformation[1] <= -0.75:
            padding = 0.2
            print('padding with 0.2')
    
    if padding is not None:
        hpadding = int(padding*H)
        wpadding = int(padding*W)
        image = np.pad(image, ((hpadding, hpadding), (wpadding, wpadding), (0, 0)), mode='reflect')

        keypoints = keypoints + torch.tensor([wpadding, hpadding])

    strain_tensor = np.array([[s[0], s[2]], 
                              [s[2], s[1]]])
    F = deformation_gradient_from_strain(strain_tensor)

    R, F_strain = polar_decomposition(F)

    # print(f'F_strain: {F_strain}')

    # Build affine transform matrix (add homogeneous coordinates)
    F_strain_h = np.eye(3)
    F_strain_h[:2, :2] = F_strain

    # Center image for proper transformation
    H,W = image.shape[:2]
    center = np.array([W,H]) / 2
    translation_to_origin = np.eye(3)
    translation_to_origin[:2, 2] = -center

    translation_back = np.eye(3)
    translation_back[:2, 2] = center

    # Combine transformations
    affine_matrix = translation_back @ F_strain_h @ translation_to_origin

    keypoints_h = np.hstack((keypoints, np.ones((keypoints.shape[0], 1))))
    transformed_keypoints = keypoints_h @ affine_matrix.T
    transformed_keypoints = transformed_keypoints[:, :2] / transformed_keypoints[:, 2:3]

    if padding is not None:
        transformed_keypoints = transformed_keypoints - np.array([wpadding, hpadding])

    return transformed_keypoints[None, :, :]

def apply_strain_to_keypoints(keypoints, s):
    strain_tensor = np.array([[s[0], s[2]], 
                              [s[2], s[1]]])

    # Transformation matrix: F = I + ε
    transformation_matrix = np.eye(2) + strain_tensor

    keypoints = np.dot(keypoints, np.linalg.inv(transformation_matrix).T)
    return keypoints