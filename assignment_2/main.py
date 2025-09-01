import numpy as np
import cv2


#  To save images to a given path
def save_image(path, image):
    if image is not None:
        cv2.imwrite(path, image)
        print("Image saved to" + path)
    else:
        print("image is not valid")


# adding reflective borders to the image, with given border width
def padding(image, border_width):
    #Adding the reflected borders
    reflect = cv2.copyMakeBorder(image,border_width,border_width,border_width,border_width, cv2.BORDER_REFLECT)
    #display the images, original and edited
    cv2.imshow('original', image)
    cv2.imshow('reflected', reflect)
    #same to solutions directory
    save_image('solutions/reflected.png', reflect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# cropping image with given values
def crop(image, x_0, x_1, y_0, y_1):  #x_0:x_1 = left-right     y_0:y_1 = top-bottom
    h, w, c = image.shape
    #modifiying the  size of the image
    cropped = image[y_0:h-y_1, x_0:w-x_1]
    print(image.shape, cropped.shape)
    cv2.imshow('original', image)
    cv2.imshow('cropped', cropped)
    save_image('solutions/cropped.png', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# resizing image with given width and height
def resize(image, width, height):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    cv2.imshow('original', image)
    cv2.imshow('resized', resized)
    save_image('solutions/resized.png', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# copying the image to an predefined picture array
def copy(image,emptyPictureArray):
    emptyPictureArray[:] = image[:]
    cv2.imshow('original', image)
    cv2.imshow('copy', emptyPictureArray)
    save_image('solutions/copy.png', emptyPictureArray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def grayscale(image):
    # transform BGR colors to graysscaled colors
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('original', image)
    cv2.imshow('grayscaled', gray)
    save_image('solutions/grayscale.png', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hsv(image):
    # transform BGR colors to HSV colors
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow('original', image)
    cv2.imshow('HSV', hsv_image)
    save_image('solutions/HSV.png', hsv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hue_shifted(image,emptyPictureArray, hue):
    emptyPictureArray[:] = image[:] #copying the image
    # Shifting the hue with limit 0 to 255
    shifted_image = np.clip(emptyPictureArray+hue, 0, 255).astype(np.uint8)

    cv2.imshow('original', image)
    cv2.imshow('shifted', shifted_image)
    save_image('solutions/shifted.png', shifted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def smoothing(image):
    #adding blur to the image with ksize = 15,15
    dst = cv2.GaussianBlur(image, (15, 15), cv2.BORDER_DEFAULT)
    cv2.imshow("Gaussian Smoothing",np.hstack((image, dst)))
    save_image('solutions/smoothing.png', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotation(image,rotation_angle):
    if rotation_angle == 90:
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('original', image)
        cv2.imshow('rotated', rotated_image)
        save_image('solutions/rotated.png', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if rotation_angle == 180:
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        cv2.imshow('original', image)
        cv2.imshow('rotated', rotated_image)
        save_image('solutions/rotated.png', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Rotation angle is not supported, only 90 or 180 degrees")
        return




def main():
    image = cv2.imread('lena.png')
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width,3), dtype=np.uint8)

    #1. Padding
    padding(image, 100)

    #2. Cropping
    crop(image,80,130,80,130)

    #3. Resize
    resize(image, 200, 200)

    #4.Manual Copy
    copy(image, emptyPictureArray)

    #.5 Grayscale
    grayscale(image)

    #6. HSV
    hsv(image)

    #7.Color Shifting
    hue_shifted(image,emptyPictureArray, 50)

    #8.Smoothing
    smoothing(image)

    #9.Rotation
    rotation(image,180)


main()
