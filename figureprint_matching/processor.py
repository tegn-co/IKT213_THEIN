import os
import time

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sift_flann import sift_flann_match_fingerprints
from orb_bf import orb_bf_match_fingerprints
from utlis import measure_performance

def process_dataset(dataset_path, results_folder,method):
    threshold = 30 #match threshold for results
    y_true, y_pred = [], []
    times, cpu_usages, ram_usages = [], [], []
    os.makedirs(results_folder, exist_ok=True)
    total_start = time.perf_counter()

    #loop through dataset
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue

        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.tif', '.png', '.jpg','jpeg'))]
        if len(image_files) != 2:
            print(f"Skipping {folder}, 2 images found {len(image_files)}")
            continue

        img1_path = os.path.join(folder_path, image_files[0])
        img2_path = os.path.join(folder_path, image_files[1])

        #match images with given method
        if method == "sift_flann":
            (match_count, match_img), perf = measure_performance(sift_flann_match_fingerprints, img1_path, img2_path)
        elif method == "orb_bf":
            (match_count, match_img), perf = measure_performance(orb_bf_match_fingerprints, img1_path, img2_path)
        else:
            raise ValueError(f"unknown method: {method}")

        times.append(perf["time_ms"])
        cpu_usages.append(perf["cpu_percent"])
        ram_usages.append(perf["ram_mb"])

        actual_match = 1 if "same" in folder.lower() else 0
        y_true.append(actual_match)

        #Classify based on matches and threshhold
        predicted_match = 1 if match_count > threshold else 0
        y_pred.append(predicted_match)
        result = "matched_"+method if predicted_match else "unmatched_"+method

        print(f"{folder}: {result.upper()} ({match_count} good matches)")

        #Save comparison image
        if match_img is not None:
            match_img_filename = f"{folder}_{result}.png"
            match_img_path = os.path.join(results_folder, match_img_filename)
            cv.imwrite(match_img_path, match_img)

    total_time = (time.perf_counter() - total_start)
    avg_time = np.mean(times) if times else 0
    avg_cpu = np.mean(cpu_usages) if cpu_usages else 0
    avg_ram = np.mean(ram_usages) if ram_usages else 0
    accuracy = accuracy_score(y_true, y_pred) * 100 if y_true else 0

    print("\n******** Resource Usage ********")
    print(f"Method: {method.upper()}")
    print(f"pairs of images: {len(times)}")
    print(f"Average time per pair: {avg_time:.2f} ms")
    print(f"Total time: {total_time:.2f} s")
    print(f"Average CPU usage: {avg_cpu:.1f}%")
    print(f"Average RAM usage: {avg_ram:.3f} MB")
    print(f"Accuracy: {accuracy:.2f}%")

    #Confusion matrix
    labels = ["Different (0)", "Same (1)"]
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="Blues", values_format="d")
    if method == "sift_flann":
        plt.title("Confusion Matrix SIFT + FLANN")
    else:
        plt.title("Confusion Matrix ORB + Brute-Force")
    plt.show()

