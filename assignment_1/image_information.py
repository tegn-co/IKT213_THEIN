import numpy as np
import cv2

def print_image_information(image):
    height, width, channels = image.shape
    print(f"A. height: {height}")
    print(f"B. width: {width}")
    print(f"C. channels: {channels}")
    print(f"D. size: {image.size}")
    print(f"E. datatype: {image.dtype}")

def main ():
    image = cv2.imread("lena.png")
    if image is None:
        print("could not read the image")
        return
    print_image_information(image)

main()