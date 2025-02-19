import cv2
import numpy as np

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

