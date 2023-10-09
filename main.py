from flask import Flask, request, render_template, Response
from subtract import alignImages
from functions import image_stream_to_cv, plot_image_grid, MyImage, getImagesFromRequest, imagesToZip
import os
import cv2 as cv
import zipfile
import io

app = Flask(__name__)

@app.route('/')
def hello(name=None):
    return render_template('index.html', name=name)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    images=[]
    if request.method == 'POST':
        images = getImagesFromRequest(request)

        #images in the images list are now in OpenCV format

        imReg, h = alignImages(images[0].image, images[1].image)

        alignedImage = MyImage()
        alignedImage.image = imReg
        alignedImage.name = 'aligned.png'
        
        images = images + [alignedImage]

        zipFile = imagesToZip(images)

        return Response(zipFile, mimetype='application/zip', headers={'Content-Disposition': 'attachment;filename=files.zip'})
    else:
        return "Hello"

if __name__ == "__main__":
   app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 5001)))