import numpy as np
import cv2
import PIL as image

def draw_keypoints(image, keypoints):
    image = image[..., ::-1]
    image = np.array(image, dtype=np.uint8)
    if type(keypoints) != np.ndarray:
        keypoints = keypoints.cpu().numpy()
    for (x,y) in keypoints:
        cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)
    return image

def visualize_keypoint_similarities(image, keypoints, similarities, min = None, max = None):
    """
    Visualizes keypoints on an image, varying color and size based on cosine similarity.
    
    Args:
        image (numpy array): The deformed image.
        keypoints (numpy array): The pixel coordinates of the deformed keypoints.
        similarities (torch.Tensor): The cosine similarities (Nx1).
    """
    image = image[..., ::-1]
    image = np.array(image, dtype=np.uint8)

    if min is None:
        min = similarities.min()
        max = similarities.max()

    # Normalize similarities to range [0,1] for color mapping
    similarities = (similarities - min) / (max - min)
    similarities = similarities.cpu().numpy()  # Convert to NumPy array if it's a tensor
    similarities = np.clip(similarities, 0, 1)  # Ensure values are within [0,1]

    # Convert similarities to a colormap (from red (low) to green (high))
    colors = (255 * np.stack((np.zeros_like(similarities) , similarities, 1 - similarities), axis=1)).astype(np.uint8)
    
    # Create visualization
    vis_image = image.copy()
    
    for (x, y), color, sim in zip(keypoints, colors, similarities):
        radius = 5  # Scale radius between 5 and 15
        cv2.circle(vis_image, (int(x), int(y)), radius, color.tolist(), thickness=-1)

    # Create color bar (gradient)
    bar_height = 30
    bar_width = vis_image.shape[1]
    color_bar = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

    for i in range(bar_width):
        value = i / bar_width  # Normalize between 0 and 1
        color = (0, int(255 * value), int(255 * (1 - value)))  # Red to Green
        color_bar[:, i] = color

    # Add labels to the color bar
    cv2.putText(color_bar, f"Low ({min})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(color_bar, f"High ({max})", (bar_width - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Stack images vertically (add color bar at bottom)
    final_vis = np.vstack((vis_image, color_bar))
    
    return final_vis, min, max