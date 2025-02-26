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
from util.Affine_Transformations import generate_strain_tensors, deformation_gradient_from_strain, polar_decomposition, apply_corotated_strain, apply_corotated_strain_with_keypoints, apply_strain, apply_strain_to_keypoints
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = get_best_device()
detector = dedode_detector_L(weights = torch.load("models/dedode_detector_L.pth", map_location = device))
descriptor = dedode_descriptor_B(weights = torch.load("models/dedode_descriptor_B.pth", map_location = device))

im_path = 'data/data/000000000042.jpg'

image = Image.open(im_path)
W,H = image.size
print(W,H)
np_image = np.array(image)

non_deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(np_image).resize((W,H)))/255.).permute(2,0,1)).float().to(device)[None]
batch = {"image": non_deformed_image}
detections = detector.detect(batch, num_keypoints = 10)
keypoints, P = detections["keypoints"], detections["confidence"]

kps1 = keypoints.cpu()
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
original_image = draw_kpts(image, kps1_pixel[0])

descriptors_1 = descriptor.describe_keypoints(batch, kps1.to(device))
print(descriptors_1['descriptions'].shape)

tensors = generate_strain_tensors()
# print(tensors)
# np_image = np.array(image)
# padded_np_image = np.array(padded_np_image)
# print(im.size)
# print(np_image.shape)
# deformed_image = apply_corotated_strain(np_image, tensors[0])
# deformed_image = apply_corotated_strain(np_image[...,::-1], tensors[0])
tensor = np.array([0.5,1.0,0.2]) #+ torch.tensor([wpadding, hpadding])
start = time.time()
deformed_image, kps2_pixel = apply_corotated_strain_with_keypoints(np_image, kps1_pixel[0], tensor, False)
end = time.time()
print(f'Time taken for transformation:{end-start}')

# kps2_pixel = kps2_pixel_full - np.array([uW, uH]) #- np.array([wpadding, hpadding])
# kps2 = detector.to_normalized_coords(torch.tensor(kps2_pixel), h, W)H
dH, dW = deformed_image.shape[:2]
kps2 = detector.to_normalized_coords(torch.tensor(kps2_pixel), dH, dW)

# kps2 = apply_strain_to_keypoints(kps1, tensors[0])
# print(kps2_pixel)
# kps2_pixel = detector.to_pixel_coords(torch.tensor(kps2), H, W)

deformed_image = np.array(deformed_image, dtype=np.uint8)
# deformed_image = deformed_padded_image[hpadding:-hpadding, wpadding:-wpadding, :]


image2 = Image.fromarray(deformed_image)
deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(deformed_image).resize((dW,dH)))/255.).permute(2,0,1)).float().to(device)[None]

# Image.fromarray(draw_kpts(image2, kps2[0])).show()
transformed_image = draw_kpts(image2, torch.tensor(kps2_pixel[0]))

# deformed_batch = {"image": inner_deformed_image}
deformed_batch = {"image": deformed_image}
descriptors_2 = descriptor.describe_keypoints(deformed_batch, kps2.float().to(device))
# print(descriptors_2['descriptions'].shape)

# print(torch.cdist(descriptors_1['descriptions'], descriptors_2['descriptions']))
# print(torch.cdist(descriptors_1['descriptions'], descriptors_2['descriptions']).shape)

normed1 = torch.nn.functional.normalize(descriptors_1['descriptions'][0], p=2, dim=1)
# print(normed1)
normed2 = torch.nn.functional.normalize(descriptors_2['descriptions'][0], p=2, dim=1)
print(torch.mm(normed1, normed2.T))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(original_image)
ax[1].imshow(transformed_image)
plt.show()