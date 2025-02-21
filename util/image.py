import numpy as np
import cv2
import PIL as image

def draw_keypoints(image, keypoints):
    image = image[..., ::-1]
    image = np.array(image, dtype=np.uint8)
    if type(keypoints) != np.ndarray:
        keypoints = keypoints.cou().numpy()
    for (x,y) in keypoints:
        cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)
    return image
