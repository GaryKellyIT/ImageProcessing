import os
import cv2
import numpy as np
import easygui
from matplotlib import pyplot as plt

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized



font = cv2.FONT_HERSHEY_SIMPLEX
# Opening an image from a file:
f = easygui.fileopenbox()
img = cv2.imread(f)

img = image_resize(img, height = 500)

cimg = img.copy()

lower_black = np.array([0,0,0])
upper_black = np.array([90,90,90])

maskblack = cv2.inRange(img, lower_black, upper_black)
cv2.imshow('maskblack', maskblack)


contours,_ = cv2.findContours(maskblack, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
hull =cv2.convexHull(contours[0])
cv2.polylines(cimg,pts=hull,isClosed=True,color=(0,255,255))


x,y,w,h=cv2.boundingRect(contours[0])
cv2.rectangle(cimg,(x,y),(x+w,y+h),(0,255,0),2)


newimg = img[y:y+h,x:x+w]

cv2.imshow('cimg', cimg)
cv2.imshow('newimg', newimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

