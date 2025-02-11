import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import warp, AffineTransform
from numpy import pad
from skimage import io
from PIL import Image
import torch
from DeDoDe import dedode_detector_L, dedode_descriptor_B
from DeDoDe.utils import get_best_device
import os
import cv2
from Affine_Transformations import generate_strain_tensors, deformation_gradient_from_strain, polar_decomposition, apply_corotated_strain, apply_corotated_strain_with_keypoints, apply_strain, apply_strain_to_keypoints


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = get_best_device()
detector = dedode_detector_L(weights = torch.load("dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("dedode_descriptor_B.pth", map_location = device))

image_dir = "data/data"
image_names = os.listdir(image_dir)
image_paths = [os.path.join(image_dir, file) for file in image_names]

# Number of points to generate
num_points = 20  # Change this as needed
half = num_points // 2

pos_values = np.logspace(np.log10(0.1), np.log10(4), num=half)
neg_values = -np.logspace(np.log10(0.1), np.log10(0.8), num=half)[::-1]

log_values = np.concatenate((neg_values, pos_values))

xvectors = np.array([[x, 0, 0] for x in log_values])
# print(xvectors)
yvectors = np.array([[0, y, 0] for y in log_values])
shearvectors = np.linspace((0,0,-0.5), (0,0,0.5), num=num_points)

x_deformation_similarities = np.zeros((num_points,))
y_deformation_similarities = np.zeros((num_points,))
shear_deformation_similarities = np.zeros((num_points,))

for j, tensor in enumerate(xvectors):

    cosines = 0.0

    for i in range(10):

        # Sample random image
        im_path = np.random.choice(image_paths)

        out1 = detector.detect_from_path(im_path, num_keypoints = 1000)
        im = Image.open(im_path)
        W,H = im.size
        kps1 = out1["keypoints"].cpu()[0]
        margin = 0.2
        condition = (np.abs(kps1[:, 0]) <= margin) & (np.abs(kps1[:, 1]) <= margin)
        kps1 = kps1[condition]
        if len(kps1) >= 1:
            kps1 = kps1[0]
            kps1 = kps1[None,...]
        else:
            kps1 = (torch.rand((1, 2)) * 2 - 1) * margin
        kps1_pixel = detector.to_pixel_coords(kps1[None,...], H, W)
        # kps1_pixel = kps1_pixel[None,...]

        descriptors_1 = descriptor.describe_keypoints_from_path(im_path, kps1[None,...].to(device))

        np_image = np.array(im)

        deformed_image, kps2_pixel = apply_corotated_strain_with_keypoints(np_image[...,::-1], kps1_pixel[0], tensor)

        kps2 = detector.to_normalized_coords(torch.tensor(kps2_pixel), H, W)

        cv2.imwrite('data/transformed_data/transformed.jpg', deformed_image)

        im_path2 = 'data/transformed_data/transformed.jpg'
        im2 = Image.open(im_path2)

        descriptors_2 = descriptor.describe_keypoints_from_path(im_path2, kps2.float().to(device))

        normed1 = torch.nn.functional.normalize(descriptors_1['descriptions'][0], p=2, dim=1)
        normed2 = torch.nn.functional.normalize(descriptors_2['descriptions'][0], p=2, dim=1)
        cosines += torch.mm(normed1, normed2.T)

    x_deformation_similarities[j] = cosines / 10

for j, tensor in enumerate(yvectors):

    cosines = 0.0

    for i in range(10):

        # Sample random image
        im_path = np.random.choice(image_paths)

        out1 = detector.detect_from_path(im_path, num_keypoints = 1000)
        im = Image.open(im_path)
        W,H = im.size
        kps1 = out1["keypoints"].cpu()[0]
        margin = 0.2
        condition = (np.abs(kps1[:, 0]) <= margin) & (np.abs(kps1[:, 1]) <= margin)
        kps1 = kps1[condition]
        if len(kps1) >= 1:
            kps1 = kps1[0]
            kps1 = kps1[None,...]
        else:
            kps1 = (torch.rand((1, 2)) * 2 - 1) * margin
        kps1_pixel = detector.to_pixel_coords(kps1[None,...], H, W)
        # kps1_pixel = kps1_pixel[None,...]

        descriptors_1 = descriptor.describe_keypoints_from_path(im_path, kps1[None,...].to(device))

        np_image = np.array(im)

        deformed_image, kps2_pixel = apply_corotated_strain_with_keypoints(np_image[...,::-1], kps1_pixel[0], tensor)

        kps2 = detector.to_normalized_coords(torch.tensor(kps2_pixel), H, W)

        cv2.imwrite('data/transformed_data/transformed.jpg', deformed_image)

        im_path2 = 'data/transformed_data/transformed.jpg'
        im2 = Image.open(im_path2)

        descriptors_2 = descriptor.describe_keypoints_from_path(im_path2, kps2.float().to(device))

        normed1 = torch.nn.functional.normalize(descriptors_1['descriptions'][0], p=2, dim=1)
        normed2 = torch.nn.functional.normalize(descriptors_2['descriptions'][0], p=2, dim=1)
        cosines += torch.mm(normed1, normed2.T)

    y_deformation_similarities[j] = cosines / 10

for j, tensor in enumerate(shearvectors):

    cosines = 0.0

    for i in range(10):

        # Sample random image
        im_path = np.random.choice(image_paths)

        out1 = detector.detect_from_path(im_path, num_keypoints = 1000)
        im = Image.open(im_path)
        W,H = im.size
        kps1 = out1["keypoints"].cpu()[0]
        margin = 0.2
        condition = (np.abs(kps1[:, 0]) <= margin) & (np.abs(kps1[:, 1]) <= margin)
        kps1 = kps1[condition]
        if len(kps1) >= 1:
            kps1 = kps1[0]
            kps1 = kps1[None,...]
        else:
            kps1 = (torch.rand((1, 2)) * 2 - 1) * margin
        kps1_pixel = detector.to_pixel_coords(kps1[None,...], H, W)
        # kps1_pixel = kps1_pixel[None,...]

        descriptors_1 = descriptor.describe_keypoints_from_path(im_path, kps1[None,...].to(device))

        np_image = np.array(im)

        deformed_image, kps2_pixel = apply_corotated_strain_with_keypoints(np_image[...,::-1], kps1_pixel[0], tensor)

        kps2 = detector.to_normalized_coords(torch.tensor(kps2_pixel), H, W)

        cv2.imwrite('data/transformed_data/transformed.jpg', deformed_image)

        im_path2 = 'data/transformed_data/transformed.jpg'
        im2 = Image.open(im_path2)

        descriptors_2 = descriptor.describe_keypoints_from_path(im_path2, kps2.float().to(device))

        normed1 = torch.nn.functional.normalize(descriptors_1['descriptions'][0], p=2, dim=1)
        normed2 = torch.nn.functional.normalize(descriptors_2['descriptions'][0], p=2, dim=1)
        cosines += torch.mm(normed1, normed2.T)

    shear_deformation_similarities[j] = cosines / 10


# Plot all results on the same graph with the vector values in the x-axis with logarithmic scale
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.title("X Deformation Similarity")
plt.plot(xvectors[:,0] + 1, x_deformation_similarities)
plt.xscale("log")
plt.xlabel("X Deformation")
plt.ylabel("Similarity")
plt.grid()

plt.subplot(1, 3, 2)
plt.title("Y Deformation Similarity")
plt.plot(yvectors[:,1] + 1, y_deformation_similarities)
plt.xscale("log")
plt.xlabel("Y Deformation")
plt.ylabel("Similarity")
plt.grid()

plt.subplot(1, 3, 3)
plt.title("Shear Deformation Similarity")
plt.plot(shearvectors[:,2], shear_deformation_similarities)
plt.xlabel("Shear Deformation")
plt.ylabel("Similarity")
plt.grid()

plt.tight_layout()

plt.show()



