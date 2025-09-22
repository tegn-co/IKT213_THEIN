import cv2 as cv
import os
import numpy as np
from matplotlib import pyplot as plt



def sobel_edge_detection(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blur = cv.GaussianBlur(image_gray, (3, 3), 0)

    sobelxy = (255 * cv.Sobel(image_blur, cv.CV_64F, dx=1, dy=1, ksize=1)).clip(0, 255).astype(np.uint8)

    cv.imshow('Sobel', sobelxy)
    cv.imwrite('solution/sobel.png', sobelxy)
    cv.waitKey(0)
    cv.destroyAllWindows()



def canny_edge_detection(image, threshold_1, threshold_2):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blur = cv.GaussianBlur(image_gray, (3, 3), 0)

    canny = cv.Canny(image_blur, threshold_1, threshold_2)
    cv.imshow('Canny', canny)
    cv.imwrite('solution/canny.png', canny)
    cv.waitKey(0)
    cv.destroyAllWindows()

def template_match(image,template):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)


    w, h = template_gray.shape[::-1]

    result = cv.matchTemplate(image_gray, template_gray, cv.TM_CCOEFF_NORMED)
    threshold = 0.9
    location = np.where(result >= threshold)

    for pt in zip(*location[::-1]):
        cv.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv.imshow('Template', image)
    cv.imwrite('solution/matched.png', image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def resize(image, scale_factor: int, up_or_down: str):

    if up_or_down == "up":
        for _ in range(scale_factor):
            image = cv.pyrUp(image)

    elif up_or_down == "down":
        for _ in range(scale_factor):
            image = cv.pyrDown(image)
    else:
        raise ValueError("Remember 'up' or 'down' as a string ")

    cv.imshow("Resized Image", image)
    cv.waitKey(0)
    cv.imwrite("solution/resized.png", image)
    cv.destroyAllWindows()




def main():
    os.makedirs("solution", exist_ok=True)
    image = cv.imread('lambo.png')
    shape = cv.imread('shapes.png')
    template = cv.imread('shapes_template.jpg')

    sobel_edge_detection(image)

    canny_edge_detection(image, 50, 50)

    template_match(shape,template)

    resize(image, 2, "down")


if __name__ == "__main__":
    main()
