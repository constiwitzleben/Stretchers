import numpy as np
import torch
from PIL import Image

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