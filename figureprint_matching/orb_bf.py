import cv2 as cv

from preprocessor import preprocess_figureprint


def orb_bf_match_fingerprints(img1_path, img2_path):
    image_1 = preprocess_figureprint(img1_path)
    image_2 = preprocess_figureprint(img2_path)

    #Intitate SIFT-detector with 2500 features
    orb = cv.ORB_create(nfeatures=2500)

    #finding keypoints og descriptors
    keypoint_1, descriptor_1 = orb.detectAndCompute(image_1, None)
    keypoint_2, descriptor_2 = orb.detectAndCompute(image_2, None)
    if descriptor_1 is None or descriptor_2 is None:
        print("No keypoints or descriptors detected")
        return 0, None

    #BF-matcher with hamming distance
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)

    #KNN
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)

    # Lowes ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    #draw only matches
    match_image = cv.drawMatches(
        image_1, keypoint_1, image_2, keypoint_2, good_matches, None,
        flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return len(good_matches), match_image




