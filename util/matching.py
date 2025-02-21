import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from util.Affine_Transformations import transform_keypoints

def draw_matches(im_A, kpts_A, im_B, kpts_B):
    im_A = np.array(im_A, dtype=np.uint8)
    im_B = np.array(im_B, dtype=np.uint8)
    kpts_A = [cv2.KeyPoint(x,y,1.) for x,y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x,y,1.) for x,y in kpts_B.cpu().numpy()]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.) for idx in range(len(kpts_A))]
    im_A, im_B = np.array(im_A), np.array(im_B)
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, 
                    matches_A_to_B, None)
    return ret

def draw_matches_with_scores(im_A, kpts_A, im_B, kpts_B, distances, threshold=5):
    im_A = np.array(im_A, dtype=np.uint8)
    im_B = np.array(im_B, dtype=np.uint8)
    kpts_A = [cv2.KeyPoint(x,y,1.) for x,y in kpts_A.cpu().numpy()]
    kpts_B = [cv2.KeyPoint(x,y,1.) for x,y in kpts_B.cpu().numpy()]

    distances = distances.cpu().numpy() if hasattr(distances, 'cpu') else np.array(distances)

    # Classify matches
    matches_good = []
    matches_bad = []
    
    for idx, distance in enumerate(distances):
        match = cv2.DMatch(idx, idx, 0.)
        if distance <= threshold:
            matches_good.append(match)  # Good match (Green)
        else:
            matches_bad.append(match)   # Bad match (Red)

    matched_img = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, matches_good, None, matchColor=(0, 0, 255))  # Green
    matched_img = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, matches_bad, matched_img, matchColor=(255, 0, 0))  # Red

    return matched_img

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