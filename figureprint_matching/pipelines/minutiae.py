import cv2
import numpy as np
import os
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from skimage.morphology import skeletonize
from skimage.filters import gabor


# ============================================================
# === 1. Forbehandling av fingeravtrykk ======================
# ============================================================

def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Kunne ikke lese bilde: {image_path}")

    # Kontrastforsterkning
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_eq = clahe.apply(img)

    # Ridge-forsterkning med Gabor-filter
    filtered = np.zeros_like(img_eq, dtype=np.float32)
    for theta in np.arange(0, np.pi, np.pi / 8):
        filt_real, _ = gabor(img_eq, frequency=0.1, theta=theta)
        filtered = np.maximum(filtered, filt_real)

    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


# ============================================================
# === 2. Ekstraksjon av minutiae =============================
# ============================================================

def extract_minutiae(binary_img):
    skel = skeletonize(binary_img // 255).astype(np.uint8)
    minutiae_points = []

    for y in range(1, skel.shape[0] - 1):
        for x in range(1, skel.shape[1] - 1):
            if skel[y, x] == 1:
                patch = skel[y - 1:y + 2, x - 1:x + 2]
                n = np.sum(patch) - 1
                if n == 1 or n >= 3:
                    minutiae_points.append([x, y])
    return np.array(minutiae_points, dtype=np.float32)


# ============================================================
# === 3. Matching og ressursmåling ===========================
# ============================================================

def match_minutiae(min1, min2, max_dist=15, ransac_thresh=5.0):
    if len(min1) < 3 or len(min2) < 3:
        return []

    # Grov matching: finn nærmeste naboer innenfor max_dist
    pairs = []
    for (x1, y1) in min1:
        dists = np.sqrt((min2[:, 0] - x1) ** 2 + (min2[:, 1] - y1) ** 2)
        idx = np.argmin(dists)
        if dists[idx] < max_dist:
            pairs.append((x1, y1, min2[idx, 0], min2[idx, 1]))

    if len(pairs) < 3:
        return []

    pts1 = np.float32([[p[0], p[1]] for p in pairs])
    pts2 = np.float32([[p[2], p[3]] for p in pairs])

    # RANSAC-estimering av affine transformasjon
    H, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)

    if inliers is None:
        return []

    # Returner kun inliers (de geometrisk konsistente matchene)
    inlier_matches = [pairs[i] for i in range(len(pairs)) if inliers[i]]
    return inlier_matches



def match_fingerprints(img1_path, img2_path):
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_mem = process.memory_info().rss / (1024**2)

    # Pipeline
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)
    min1 = extract_minutiae(img1)
    min2 = extract_minutiae(img2)
    matches = match_minutiae(min1, min2)

    elapsed = (time.perf_counter() - start_time) * 1000
    mem_used = process.memory_info().rss / (1024**2) - start_mem

    # Visualiser maks 200 matcher
    vis_img = visualize_matches(img1, img2, matches[:200])
    return len(matches), elapsed, mem_used, vis_img


# ============================================================
# === 4. Visualisering =======================================
# ============================================================

def visualize_matches(img1, img2, matches):
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    combined = np.hstack((img1_rgb, img2_rgb))
    offset_x = img1.shape[1]

    for (x1, y1, x2, y2) in matches:
        cv2.line(combined, (int(x1), int(y1)), (int(x2) + offset_x, int(y2)), (0,255,0), 1)
        cv2.circle(combined, (int(x1), int(y1)), 1, (0,0,255), -1)
        cv2.circle(combined, (int(x2) + offset_x, int(y2)), 1, (255,0,0), -1)
    return combined


# ============================================================
# === 5. Kjøring på dataset ==================================
# ============================================================

def process_dataset(dataset_path, results_folder):
    threshold = 50  # juster basert på empiriske tester
    y_true, y_pred = [], []
    os.makedirs(results_folder, exist_ok=True)

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.png', '.jpg'))]
        if len(image_files) != 2:
            print(f"Skipping {folder}: expected 2 images, found {len(image_files)}")
            continue

        img1_path = os.path.join(folder_path, image_files[0])
        img2_path = os.path.join(folder_path, image_files[1])

        match_count, time_ms, mem_MB, vis_img = match_fingerprints(img1_path, img2_path)

        actual_match = 1 if "same" in folder.lower() else 0
        y_true.append(actual_match)
        predicted_match = 1 if match_count > threshold else 0
        y_pred.append(predicted_match)

        result = "matched" if predicted_match else "unmatched"
        print(f"{folder}: {result.upper()} ({match_count} matches, {time_ms:.1f} ms, {mem_MB:.2f} MB)")

        if vis_img is not None:
            save_path = os.path.join(results_folder, f"{folder}_{result}.png")
            cv2.imwrite(save_path, vis_img)

    # Confusion Matrix
    labels = ["Different (0)", "Same (1)"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(6,5))
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Minutiae Matching)")
    plt.show()
