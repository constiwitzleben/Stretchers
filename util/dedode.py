import numpy as np
import torch
from PIL import Image
from util.Affine_Transformations import apply_corotated_strain_with_keypoints, generate_strain_tensors, generate_27_strain_tensors

def detect_and_describe(im, detector, descriptor, device, num_keypoints = 100):
    if im.dtype != np.uint8:
        im = np.array(im,dtype=np.uint8)
    h, w = im.shape[:2]
    im = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(im).resize((w,h)))/255.).permute(2,0,1)).float().to(device)[None]
    batch = {"image": im}
    detections = detector.detect(batch, num_keypoints = num_keypoints)
    keypoints, P = detections["keypoints"], detections["confidence"]
    descriptions = descriptor.describe_keypoints(batch, keypoints.to(device))["descriptions"].squeeze()
    return keypoints, P, descriptions

def get_affine_deformed_descriptions(image, pixel_keypoints, tensors, detector, descriptor, device):

    H, W = image.shape[:2]

    affine_deformed_descriptions = []

    # Loop over deformations
    for j, deformation in enumerate(tensors):
        
        # Apply deformation
        deformed_image, deformed_pixel_keypoints = apply_corotated_strain_with_keypoints(image, pixel_keypoints, deformation, dataset_mode=False)
        dH, dW = deformed_image.shape[:2]
        deformed_keypoints = detector.to_normalized_coords(torch.tensor(deformed_pixel_keypoints), dH, dW).to(torch.float32)[0]

        # Get deformed descriptor without saving and reading the image
        deformed_image = np.array(deformed_image, dtype=np.uint8)
        deformed_image = descriptor.normalizer(torch.from_numpy(np.array(Image.fromarray(deformed_image).resize((dW,dH)))/255.).permute(2,0,1)).float().to(device)[None]
        deformed_batch = {"image": deformed_image}
        deformed_descriptions = descriptor.describe_keypoints(deformed_batch, deformed_keypoints[None,...].to(device))["descriptions"].squeeze()
        affine_deformed_descriptions.append(deformed_descriptions.cpu())
    
    return np.array(affine_deformed_descriptions)
    