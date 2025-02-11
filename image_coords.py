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

im_path = 'data/data/000000000042.jpg'
# H = 784
# W = 784
# image = torch.from_numpy(np.array(Image.open(im_path).resize((W,H)))/255.).permute(2,0,1).float()
# plt.imshow(image.permute(1, 2, 0))
# plt.show()

image = Image.open(im_path)
W,H = image.size
print(W,H)
np_image = np.array(image)

hpadding = 3*H
wpadding = 3*W
padded_np_image = np.pad(np_image, ((hpadding, hpadding), (wpadding, wpadding), (0, 0)), mode='reflect')
print(padded_np_image.shape)

inner_cutoff = 2
double_cutoff = inner_cutoff*2
uH = (inner_cutoff - 1) * H // double_cutoff
lH = (inner_cutoff + 1) * H // double_cutoff
uW = (inner_cutoff - 1) * W // double_cutoff
lW = (inner_cutoff + 1) * W // double_cutoff

np_inner_image = np_image[uH:lH, uW:lW, :]
h, w = np_inner_image.shape[:2]
print(h,w)
inner_image = Image.fromarray(np_inner_image)

np_inner_image = np.array(np_inner_image,dtype=np.uint8)
non_deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(np_inner_image).resize((w,h)))/255.).permute(2,0,1)).float().to(device)[None]
batch = {"image": non_deformed_image}
detections = detector.detect(batch, num_keypoints = 1000)
keypoints, P = detections["keypoints"], detections["confidence"]

kps1 = keypoints.cpu()[0]
# print(kps1)
margin = 0.2
condition = (np.abs(kps1[:, 0]) <= margin) & (np.abs(kps1[:, 1]) <= margin)
kps1 = kps1[condition]
kps1 = kps1[:8]
# print(kps1)
kps1_pixel = detector.to_pixel_coords(kps1[None,...], h, w)

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
original_image = draw_kpts(inner_image, kps1_pixel[0])

descriptors_1 = descriptor.describe_keypoints(batch, kps1[None,...].to(device))
print(descriptors_1['descriptions'].shape)

tensors = generate_strain_tensors()
# print(tensors)
np_image = np.array(image)
padded_np_image = np.array(padded_np_image)
# print(im.size)
# print(np_image.shape)
# deformed_image = apply_corotated_strain(np_image, tensors[0])
# deformed_image = apply_corotated_strain(np_image[...,::-1], tensors[0])
tensor = np.array([-0.5,-0.5,0.4])
kps1_pixel_full = kps1_pixel[0] + torch.tensor([uW, uH]) + torch.tensor([wpadding, hpadding])
deformed_padded_image, kps2_pixel_full = apply_corotated_strain_with_keypoints(padded_np_image, kps1_pixel_full, tensor)
# print(tensors[22])

kps2_pixel = kps2_pixel_full - np.array([uW, uH]) - np.array([wpadding, hpadding])
kps2 = detector.to_normalized_coords(torch.tensor(kps2_pixel), h, w)


# kps2 = apply_strain_to_keypoints(kps1, tensors[0])
# print(kps2_pixel)
# kps2_pixel = detector.to_pixel_coords(torch.tensor(kps2), H, W)



deformed_padded_image = np.array(deformed_padded_image, dtype=np.uint8)
deformed_image = deformed_padded_image[hpadding:-hpadding, wpadding:-wpadding, :]
inner_deformed_image = deformed_image[uH:lH, uW:lW, :]
im2 = Image.fromarray(inner_deformed_image)
inner_deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(inner_deformed_image).resize((w,h)))/255.).permute(2,0,1)).float().to(device)[None]

# Image.fromarray(draw_kpts(im2, kps2[0])).show()
transformed_image = draw_kpts(im2, torch.tensor(kps2_pixel[0]))

deformed_batch = {"image": inner_deformed_image}
descriptors_2 = descriptor.describe_keypoints(deformed_batch, kps2.float().to(device))
print(descriptors_2['descriptions'].shape)

print(torch.cdist(descriptors_1['descriptions'], descriptors_2['descriptions']))
print(torch.cdist(descriptors_1['descriptions'], descriptors_2['descriptions']).shape)

normed1 = torch.nn.functional.normalize(descriptors_1['descriptions'][0], p=2, dim=1)
print(normed1)
normed2 = torch.nn.functional.normalize(descriptors_2['descriptions'][0], p=2, dim=1)
print(torch.mm(normed1, normed2.T))

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(original_image)
ax[1].imshow(transformed_image)
plt.show()

