import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.03

def plot_image_grid(images, ncols=None, cmap='gray'):
    '''Plot a grid of images'''
    if not ncols:
        factors = [i for i in range(1, len(images)+1) if len(images) % i == 0]
        ncols = factors[len(factors) // 2] if len(factors) else len(images) // 4 + 1
    nrows = int(len(images) / ncols) + int(len(images) % ncols)
    imgs = [images[i] if len(images) > i else None for i in range(nrows * ncols)]
    f, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2*nrows))
    axes = axes.flatten()[:len(imgs)]
    for img, ax in zip(imgs, axes.flatten()): 
        if np.any(img):
            if len(img.shape) > 2 and img.shape[2] == 1:
                img = img.squeeze()
            ax.imshow(img, cmap=cmap)


def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
  im2Gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
 
  # Detect ORB features and compute descriptors.
  orb = cv.ORB_create(MAX_FEATURES, 1.01, 1)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
 
  # Match features.
  matcher = cv.DescriptorMatcher_create(cv.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
 
  # Sort matches by score
  #matches.sort(key=lambda x: x.distance, reverse=False)
  matches = sorted(matches, key = lambda x:x.distance)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  #imMatches = cv.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  #cv.imwrite("matches.jpg", imMatches)
  #plt.imshow(imMatches)
  #plt.show()

  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
 
  # Find homography
  #h, mask = cv.findHomography(points1, points2, cv.RANSAC)

  # estimateAffinePartial2D
  h, inliners = cv.estimateAffinePartial2D(points1, points2, cv.RANSAC)

  print(h)
 
  # Use affine
  height, width, channels = im2.shape
  im1Reg = cv.warpAffine(im1, h, (width, height))
 
  return im1Reg, h

def threshold(img1):
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img1.copy()
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.medianBlur(img,5)
    ret,th1 = cv.threshold(img,32,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    images = [img, th1, th2, th3]
    #plot_image_grid(images)
    return th1

def getColorFromImage(img, channel):
    channel_image = img.copy()

    match channel:
        case 'r':
            channel_image[:,:,1] = 0#channel_image[:,:,0]
            channel_image[:,:,2] = 0#channel_image[:,:,0]
        case 'g':
            channel_image[:,:,0] = 0#channel_image[:,:,1]
            channel_image[:,:,2] = 0#channel_image[:,:,1]
        case 'b':
            channel_image[:,:,0] = 0#channel_image[:,:,2]
            channel_image[:,:,1] = 0#channel_image[:,:,2]

    return channel_image;
    return cv.cvtColor(channel_image, cv.COLOR_BGR2GRAY)

def main():

    b = cv.imread('resources/6065t.jpg')
    a = cv.imread('resources/8066t-fout-rotation.jpeg')
    
    c, h  =alignImages(a, b)

    plot_image_grid([a, b, c])

    plt.show()

    return


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()