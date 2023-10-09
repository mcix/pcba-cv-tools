import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from functions import threshold, plot_image_grid

def main():
  top = cv.imread('resources/AVANPR21/6598t.jpg')
  bottom = cv.imread('resources/AVANPR21/6598b.jpg')

  captch_ex(top)

  a = cv.cvtColor(top, cv.COLOR_BGR2GRAY)

  a = threshold(a)

  plot_image_grid(a)

  plt.show()

# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
  main()