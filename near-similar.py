from sentence_transformers import SentenceTransformer, util
from PIL import Image
import glob
import os
import cv2 as cv
import numpy as np

from align import alignImages

def initModel():
    # Load the OpenAI CLIP Model
    print('Loading CLIP Model...')
    model = SentenceTransformer('clip-ViT-B-32')
    return model

def similarScore(img1, img2, model):
    # Next we compute the embeddings
    # To encode an image, you can use the following code:
    # from PIL import Image
    # encoded_image = model.encode(Image.open(filepath))
    #image_names = list(glob.glob('./*.jpg'))
    #print("Images:", len(image_names))
    encoded_image = model.encode([img1, img2], batch_size=2, convert_to_tensor=False, show_progress_bar=False)

    # Now we run the clustering algorithm. This function compares images aganist 
    # all other images and returns a list with the pairs that have the highest 
    # cosine similarity score
    processed_images = util.paraphrase_mining_embeddings(encoded_image)
    NUM_SIMILAR_IMAGES = 10 

    return processed_images[0][0]

    # =================
    # DUPLICATES
    # =================
    print('Finding duplicate images...')
    # Filter list for duplicates. Results are triplets (score, image_id1, image_id2) and is scorted in decreasing order
    # A duplicate image will have a score of 1.00
    # It may be 0.9999 due to lossy image compression (.jpg)
    duplicates = [image for image in processed_images if image[0] >= 0.999]

    # Output the top X duplicate images
    for score, image_id1, image_id2 in duplicates[0:NUM_SIMILAR_IMAGES]:
        print("\nScore: {:.3f}%".format(score * 100))
        #print(image_names[image_id1])
        #print(image_names[image_id2])

    # =================
    # NEAR DUPLICATES
    # =================
    print('Finding near duplicate images...')
    # Use a threshold parameter to identify two images as similar. By setting the threshold lower, 
    # you will get larger clusters which have less similar images in it. Threshold 0 - 1.00
    # A threshold of 1.00 means the two images are exactly the same. Since we are finding near 
    # duplicate images, we can set it at 0.99 or any number 0 < X < 1.00.
    threshold = 0.99
    near_duplicates = [image for image in processed_images if image[0] < threshold]

    for score, image_id1, image_id2 in near_duplicates[0:NUM_SIMILAR_IMAGES]:
        print("\nScore: {:.3f}%".format(score * 100))
        #print(image_names[image_id1])
        #print(image_names[image_id2])

def divide_img_blocks(img, n_blocks=(2,2)):
   horizontal = np.array_split(img, n_blocks[0])
   splitted_img = [np.array_split(block, n_blocks[1], axis=1) for block in horizontal]
   return np.asarray(splitted_img, dtype=np.ndarray).reshape(n_blocks)

def cv2toPil(opencv_image):
    color_coverted = cv.cvtColor(opencv_image, cv.COLOR_BGR2RGB) 
    return Image.fromarray(color_coverted) 

def main():
    model = initModel()

    a = cv.imread('resources/6065t.png')
    b = cv.imread('resources/8066t-fout.png')

    c, h  =alignImages(a, b)

    blocksize = 10
    a_blocks = divide_img_blocks(b, (blocksize,blocksize))
    b_blocks =  divide_img_blocks(c, (blocksize,blocksize))

    height, width, channels = a.shape
    overlay = np.zeros((height,width,3), np.uint8)

    
        
    for i, imgList in enumerate(a_blocks):
        for j, img in enumerate(imgList):
            img2 = b_blocks[i][j]
            #cv.imshow('a', img)
            #cv.imshow('b', img2)
            #cv.waitKey()
            score = similarScore(cv2toPil(img), cv2toPil(img2), model)
            print("Score: {} {} {:.3f}%".format(i,j,score * 100))

            height, width, channels = img.shape 

            w = width
            h = height
            x, y, w, h = w * j, h * i, w-1, h-1  # Rectangle parameters

            dScore = min(255, 255 * ((1 - score) * 10));
            cv.rectangle(overlay, (x, y), (x+w, y+h), (0, 0 , dScore), -1)  # A filled rectangle

    alpha = 0.8  # Transparency factor.

    image_new = cv.addWeighted(overlay, alpha, a, 1 - alpha, 0)

    cv.imshow( 'overlay', image_new)
    cv.waitKey()


    return


# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
    main()