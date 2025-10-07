import cv2 as cv


def preprocess_figureprint(image_path):
    img = cv.imread(image_path)
    #cv.imshow("original",img)

    if img is None:
        raise FileNotFoundError(f"Kunne ikke lese bilde: {image_path}")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow("gray",img)
    _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    #cv.imshow("bin",img_bin)
    #cv.waitKey(0)
    return img_bin

