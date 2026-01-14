import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from .affine_transformations import transform_keypoints
from .fenics_deformation import get_strain

def draw_matches(im_A, kpts_A, im_B, kpts_B):
    im_A = np.array(im_A, dtype=np.uint8)
    im_B = np.array(im_B, dtype=np.uint8)
    kpts_A = [cv2.KeyPoint(x, y, 1.) for x, y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x, y, 1.) for x, y in kpts_B.cpu().numpy()]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(len(kpts_A))]

    # Draw matches with green lines and increased thickness
    ret = cv2.drawMatches(
        im_A, kpts_A, im_B, kpts_B, matches_A_to_B, None, 
        matchColor=(0, 255, 0),  # Green color (BGR format)
        singlePointColor=None,   # Default keypoint color
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return ret

def draw_matches_with_scores(im_A, kpts_A, im_B, kpts_B, distances, threshold=5, lines=True):
    im_A = np.array(im_A, dtype=np.uint8)
    im_B = np.array(im_B, dtype=np.uint8)
    
    # Convert keypoints from tensors to OpenCV KeyPoints
    kpts_A = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts_B.cpu().numpy()]
    
    # Stack images horizontally for visualization
    hA, wA = im_A.shape[:2]
    hB, wB = im_B.shape[:2]
    out_image = np.ones((max(hA, hB), wA + wB, 3), dtype=np.uint8)*255
    out_image[:hA, :wA] = im_A
    out_image[:hB, wA:] = im_B
    
    # Draw matches manually with colored lines
    for i, (kpA, kpB, dist) in enumerate(zip(kpts_A, kpts_B, distances)):
        xA, yA = int(kpA.pt[0]), int(kpA.pt[1])
        xB, yB = int(kpB.pt[0]) + wA, int(kpB.pt[1])  # Shift x-coordinates for B
        
        # Choose color based on threshold
        color = (0, 255, 0) if dist < threshold else (0, 0, 255)  # Green if below, Red if above
        
        # Draw the match line
        if lines is True:
            if dist < threshold:
                cv2.line(out_image, (xA, yA), (xB, yB), color, 2)
        
        # Draw keypoints
        cv2.circle(out_image, (xA, yA), 4, color, -1)  # Blue keypoints on A
        cv2.circle(out_image, (xB, yB), 4, color, -1)  # Blue keypoints on B

    return out_image

def draw_matching_comparison(img_baseline, img_stretched, image_dir):

    # Ensure both images have the same height
    w_baseline, h_baseline = img_baseline.size
    w_stretched, h_stretched = img_stretched.size
    new_width = max(w_baseline, w_stretched)
    new_height = h_baseline + h_stretched + 20  # Adding a margin between images

    # Create a blank canvas
    combined_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
    combined_img.paste(img_baseline, (0, 0))
    combined_img.paste(img_stretched, (0, h_baseline + 20))  # Add a margin

    # Add labels
    draw = ImageDraw.Draw(combined_img)
    font = ImageFont.load_default()  # You can replace this with a custom font if needed
    draw.text((10, 10), "Baseline Matches", fill=(0, 0, 0), font=font)
    draw.text((10, h_baseline + 30), "Stretched Matches", fill=(0, 0, 0), font=font)

    # Save or display the combined image
    combined_img.save(image_dir)
    # combined_img.show()

def print_matching_accuracy(base_matches, deformed_matches, np_image, deformation, uW, uH, threshold=5):
    whole_pixel_stretched_matches = base_matches.cpu() + torch.tensor([uW, uH])
    whole_pixel_gt_deformed_matches = transform_keypoints(np_image, whole_pixel_stretched_matches, deformation)
    pixel_gt_deformed_keypoints = whole_pixel_gt_deformed_matches - np.array([uW, uH])

    distances = (deformed_matches.cpu() - pixel_gt_deformed_keypoints).norm(dim=2)[0]
    good = (distances < threshold).sum().item()
    total = len(distances)
    accuracy = good / total
    print(f'Accuracy: {accuracy} ({good} / {total})')

def strain_entropy(s,keypoints,W,H):
    if len(keypoints) == 0:
        return 0
    strain_values = np.array([get_strain(s, pixel, W, H, 10, 10) for pixel in keypoints])
    all_strain_values = s.vector()[:]
    partitions = np.quantile(all_strain_values, np.linspace(0, 1, 11))
    hist, bin_edges = np.histogram(strain_values, bins=partitions, density=False)
    hist = hist / hist.sum()
    hist += 1e-10
    entropy = -np.sum(hist*np.log(hist))
    entropy /= np.log(10)  # Normalize by log
    return entropy
    
def strain_balanced_precision(s, keypoints, good_mask, W, H):
    strain_values = np.array([get_strain(s, pixel, W, H, 10, 10) for pixel in keypoints])
    # Partition the keypoints into 10 partitions by strain
    all_strain_values = s.vector()[:]
    partitions = np.quantile(all_strain_values, np.linspace(0, 1, 11))
    balanced_precision = 0
    for i in range(10):
        partition = partitions[i:i+2]
        partition_mask = (strain_values >= partition[0]) & (strain_values < partition[1])
        good_partition = good_mask & partition_mask
        if good_partition.sum() > 0:
            balanced_precision += good_partition.sum() / partition_mask.sum()
            # print(good_partition.sum() / partition_mask.sum())
    balanced_precision /= 10
    return balanced_precision