import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def main():
    imgL = cv.imread('resources/up-l.jpg', cv.IMREAD_GRAYSCALE)
    imgR = cv.imread('resources/up-r.jpg', cv.IMREAD_GRAYSCALE)
    stereo = cv.StereoBM_create(numDisparities=640, blockSize=15)
    disparity = stereo.compute(imgL,imgR)
    plt.imshow(disparity,'gray')
    plt.show()

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()