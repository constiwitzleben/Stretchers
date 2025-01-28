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

im_path = 'data/data/000000000042.jpg'
# H = 784
# W = 784
# image = torch.from_numpy(np.array(Image.open(im_path).resize((W,H)))/255.).permute(2,0,1).float()
# plt.imshow(image.permute(1, 2, 0))
# plt.show()

out1 = detector.detect_from_path(im_path, num_keypoints = 6)
im = Image.open(im_path)
W,H = im.size
# print(W,H)
kps1 = out1["keypoints"].cpu()
print(kps1)
kps1_pixel = detector.to_pixel_coords(kps1, H, W)

def draw_kpts(im, kpts):    
    kpts = [cv2.KeyPoint(x,y,15.) for x,y in kpts.cpu().numpy()]
    im = np.array(im)
    ret = cv2.drawKeypoints(im, kpts, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return ret

# plt.imshow(draw_kpts(Image.open(im_path), kps[0]))
# plt.show()

# Image.fromarray(draw_kpts(im, kps1_pixel[0])).show()
# test_kpts = torch.tensor([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
# test_kpts_pixel = detector.to_pixel_coords(test_kpts, H, W)
# original_image = draw_kpts(im, test_kpts_pixel)
original_image = draw_kpts(im, kps1_pixel[0])

tensors = generate_strain_tensors()
np_image = np.array(im)
# print(im.size)
# print(np_image.shape)
# deformed_image = apply_corotated_strain(np_image, tensors[0])
# deformed_image = apply_corotated_strain(np_image[...,::-1], tensors[0])
deformed_image, kps2_pixel = apply_corotated_strain_with_keypoints(np_image[...,::-1], kps1_pixel[0], tensors[0])

kps2 = detector.to_normalized_coords(torch.tensor(kps2_pixel), H, W)
print(kps2)

# kps2 = apply_strain_to_keypoints(kps1, tensors[0])
# print(kps2_pixel)
# kps2_pixel = detector.to_pixel_coords(torch.tensor(kps2), H, W)
cv2.imwrite('data/transformed_data/000000000042.jpg', deformed_image)

im_path2 = 'data/transformed_data/000000000042.jpg'
im2 = Image.open(im_path2)

# Image.fromarray(draw_kpts(im2, kps2[0])).show()
transformed_image = draw_kpts(im2, torch.tensor(kps2_pixel[0]))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(original_image)
ax[1].imshow(transformed_image)
plt.show()

