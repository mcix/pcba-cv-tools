# Get started
Create a virtual environment, active it, and then install the requirements:

    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

## align.py [working]
Aligns 2 images, only rotates them and scales them. It does not translate them. Preparation from 'Wapper' or computer vision comparison

## stitch.py [working]
Uses OpenCV Stitcher to stich images together (like a panorama function)

## structuralsimilarityindex.py [work in progress]
This example tries to find mismatches in the 2 images
First it aligns the images using align.py

## main.py [work in progress]
Project to expose computer vision tools using a restfull interface.

## Notes
To compare images:
https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv

