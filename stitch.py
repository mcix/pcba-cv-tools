import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def stitch(imgs):

  stitchy=cv.Stitcher.create()
  (dummy,output)=stitchy.stitch(imgs)

  cv.imwrite('stitch.jpg', output)

  plt.imshow(output)
  plt.show()

def main():
  a = cv.imread('resources/panorama/10312.jpg')
  b = cv.imread('resources/panorama/10313.jpg')
  c = cv.imread('resources/panorama/10314.jpg')
  d = cv.imread('resources/panorama/10315.jpg')

  stitch(imgs=[a,b,c,d])

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()