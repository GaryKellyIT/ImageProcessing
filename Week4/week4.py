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
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# color range
lower_red1 = np.array([0,170,100])
upper_red1 = np.array([10,255,255])
lower_red2 = np.array([160,170,100])
upper_red2 = np.array([180,255,255])
lower_green = np.array([40,140,90])
upper_green = np.array([90,255,255])
# lower_yellow = np.array([15,100,100])
# upper_yellow = np.array([35,255,255])
lower_yellow = np.array([15,210,170])#lower_yellow = np.array([15,150,150])
upper_yellow = np.array([35,255,255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
maskg = cv2.inRange(hsv, lower_green, upper_green)
masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
maskr = cv2.add(mask1, mask2)
cv2.imshow('img', img)
##cv2.imshow('maskg', maskg)
##cv2.imshow('masky', masky)
##cv2.imshow('maskr', maskr)
##cv2.imshow("g", np.hstack([img, maskg]))
##cv2.imshow("r", np.hstack([img, maskr]))
##cv2.imshow("y", np.hstack([img, masky]))

redresult = cv2.bitwise_and(img, img, mask=maskr)
yellowresult = cv2.bitwise_and(img, img, mask=masky)
greenresult = cv2.bitwise_and(img, img, mask=maskg)



img_gray_r = cv2.cvtColor(redresult, cv2.COLOR_HSV2RGB)
img_gray_r = cv2.cvtColor(img_gray_r, cv2.COLOR_RGB2GRAY)
img_gray_y = cv2.cvtColor(yellowresult, cv2.COLOR_HSV2RGB)
img_gray_y = cv2.cvtColor(img_gray_y, cv2.COLOR_RGB2GRAY)
img_gray_g = cv2.cvtColor(greenresult, cv2.COLOR_HSV2RGB)
img_gray_g = cv2.cvtColor(img_gray_g, cv2.COLOR_RGB2GRAY)


img_gray_r = cv2.GaussianBlur(img_gray_r, (5, 5), 0)
img_gray_g = cv2.GaussianBlur(img_gray_g, (5, 5), 0)
img_gray_y = cv2.GaussianBlur(img_gray_y, (5, 5), 0)

cv2.imshow('maskg', img_gray_g)
cv2.imshow('masky', img_gray_y)
cv2.imshow('maskr', img_gray_r)


size = img.shape
# print size

# hough circle detect


r_circles = cv2.HoughCircles(img_gray_r, cv2.HOUGH_GRADIENT, 1, 80, param1 = 50, param2 = 18, minRadius = 25, maxRadius = 70)

g_circles = cv2.HoughCircles(img_gray_g, cv2.HOUGH_GRADIENT, 1, 80, param1 = 50, param2 = 18, minRadius = 25, maxRadius = 70)

y_circles = cv2.HoughCircles(img_gray_y, cv2.HOUGH_GRADIENT, 1, 80, param1 = 50, param2 = 18, minRadius = 25, maxRadius = 70)

# traffic light detect
r = 5
bound = 4.0 / 10
if r_circles is not None:
    r_circles = np.round(r_circles[0, :]).astype("int")
 
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in r_circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(cimg, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(cimg, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
		
if g_circles is not None:
    g_circles = np.round(g_circles[0, :]).astype("int")
 
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in g_circles:
        
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(cimg, (x, y), r, (0, 255, 0), 4)
        
		
if y_circles is not None:
    y_circles = np.round(y_circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in y_circles:
        
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(cimg, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(cimg, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

if r_circles is None and g_circles is None and y_circles is None:
    print("No circles found")
elif r_circles is not None:
    rsize = int(r_circles.size /3)
    if rsize == 1:
        print("{} red circle found".format(rsize))
    else:
        print("{} red circles found".format(rsize))
elif g_circles is not None:
    gsize = int(g_circles.size /3)
    if gsize == 1:
        print("{} green circle found".format(gsize))
    else:
        print("{} green circles found".format(gsize))
elif y_circles is not None:
    ysize = int(y_circles.size /3)
    if ysize == 1:
        print("{} yellow circle found".format(ysize))
    else:
        print("{} yellow circles found".format(ysize))
   
    
    
cv2.imshow('detected results', np.hstack([img,cimg]))

cv2.waitKey(0)
cv2.destroyAllWindows()


