import cv2 as cv
import numpy as np
import zipfile
import io
from matplotlib import pyplot as plt
from werkzeug.utils import secure_filename

class MyImage:
  name = ''
  image = 0

def getImagesFromRequest(request):
    images = []
    files = request.files.getlist("file")
    for f in files:
        img = MyImage()
        img.name = secure_filename(f.filename)
        img.image = image_stream_to_cv(f.read())
        images = images + [img]
    return images

def imagesToZip(images):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for i in images:
            retval, buf = cv.imencode(i.name, i.image)
            zip_file.writestr(i.name, buf)
    return zip_buffer.getvalue()

def cv_to_img(cv_img):
    retval, buf = cv.imencode('.png', cv_img)
    return buf

def threshold(img1):
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img1.copy()
    assert img is not None, "file could not be read, check with os.path.exists()"
    img = cv.medianBlur(img,5)
    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    images = [img, th1, th2, th3]
    #plot_image_grid(images)
    return images

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

def image_stream_to_cv(in_stream):
    img = cv.imdecode(np.fromstring(in_stream, np.uint8), cv.IMREAD_UNCHANGED)
    return img