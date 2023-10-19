from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import cv2 as cv
import numpy as np

from align import alignImages, plot_image_grid

def structuralSimilarityIndex3(image1, image2):
    # Load images as grayscale

    # Calculate the per-element absolute difference between 
    # two arrays or between an array and a scalar
    diff = 255 - cv.absdiff(image1, image2)

    cv.imshow('diff', diff)
    cv.waitKey()

def structuralSimilarityIndex2(image1, image2):
    # Load images as grayscale

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(image1, image2, full=True, channel_axis=2)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] image1 we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    print("Image Similarity: {:.4f}%".format(score * 100))

    cv.imshow('diff', diff)
    cv.waitKey()

def structuralSimilarityIndex(first, second):

    # Convert images to grayscale
    first_gray = cv.cvtColor(first, cv.COLOR_BGR2GRAY)
    second_gray = cv.cvtColor(second, cv.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    score, diff = structural_similarity(first_gray, second_gray, full=True)
    print("Similarity Score: {:.3f}%".format(score * 100))

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type so we must convert the array 
    # to 8-bit unsigned integers in the range [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image, followed by finding contours to
    # obtain the regions that differ between the two images
    thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Highlight differences
    mask = np.zeros(first.shape, dtype='uint8')
    filled = second.copy()

    for c in contours:
        area = cv.contourArea(c)
        if area > 100:
            x,y,w,h = cv.boundingRect(c)
            cv.rectangle(first, (x, y), (x + w, y + h), (36,255,12), 2)
            cv.rectangle(second, (x, y), (x + w, y + h), (36,255,12), 2)
            cv.drawContours(mask, [c], 0, (0,255,0), -1)
            cv.drawContours(filled, [c], 0, (0,255,0), -1)

    cv.imshow('first', first)
    cv.imshow('second', second)
    cv.imshow('diff', diff)
    cv.imshow('mask', mask)
    cv.imshow('filled', filled)
    cv.waitKey()

def main():

    b = cv.imread('resources/6065t.png')
    a = cv.imread('resources/8066t-fout.png')
    
    c, h  =alignImages(a, b)

    #cv.medianBlur(a, 9, a);
    #cv.medianBlur(b, 9, b);
    #cv.medianBlur(c, 9, c);

    #plot_image_grid([a, b, c])

    #plt.show()

    structuralSimilarityIndex(b, c)

    return


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()