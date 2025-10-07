import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def preprocess_fingerprint(image_path):
    """Les og binariser et fingeravtrykk med Otsu-invertering."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Kunne ikke lese bilde: {image_path}")
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin


def match_fingerprints(img1_path, img2_path):
    """Matcher fingeravtrykk ved bruk av SIFT + FLANN."""
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)

    # Initialiser SIFT-detektor
    sift = cv2.SIFT_create(nfeatures=1000)

    # Finn keypoints og descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None

    # FLANN-parametere (KD-tree for SIFT)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN-match
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe’s ratio-test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # Tegn kun faktiske matcher
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return len(good_matches), match_img


def process_dataset(dataset_path, results_folder):
    """Kjører matching over alle mapper og viser confusion matrix."""
    threshold = 20  # juster basert på testresultater
    y_true, y_pred = [], []

    os.makedirs(results_folder, exist_ok=True)

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.tif', '.png', '.jpg'))]
        if len(image_files) != 2:
            print(f"Skipping {folder}: expected 2 images but found {len(image_files)}")
            continue

        img1_path = os.path.join(folder_path, image_files[0])
        img2_path = os.path.join(folder_path, image_files[1])

        match_count, match_img = match_fingerprints(img1_path, img2_path)

        # Sann label (1 for same, 0 for different)
        actual_match = 1 if "same" in folder.lower() else 0
        y_true.append(actual_match)

        # Predikert label
        predicted_match = 1 if match_count > threshold else 0
        y_pred.append(predicted_match)
        result = "sift_flann_matched" if predicted_match else "sift_flann_unmatched"

        print(f"{folder}: {result.upper()} ({match_count} good matches)")

        # Lagre bilde
        if match_img is not None:
            out_name = f"{folder}_{result}.png"
            cv2.imwrite(os.path.join(results_folder, out_name), match_img)

    # Confusion Matrix
    labels = ["Different (0)", "Same (1)"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - SIFT + FLANN")
    plt.show()


# === Kjøring ===
if __name__ == "__main__":
    dataset_path = r"C:\Users\dunk1\ikt213\IKT213_THEIN\figureprint_matching\data"
    results_folder = r"C:\Users\dunk1\ikt213\IKT213_THEIN\figureprint_matching\results_sift"
    process_dataset(dataset_path, results_folder)
