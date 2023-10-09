import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def main():
    img = cv.imread('resources/a.png')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray,50,0.01,10)
    corners = np.int0(corners)
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),3,255,-1)
    plt.imshow(img),plt.show()

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()