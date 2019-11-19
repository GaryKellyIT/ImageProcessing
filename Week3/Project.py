import numpy as np
import cv2
from matplotlib import pyplot as plt
import easygui

#change to user picked image
I = cv2.imread("../input/GreenLight.jpg")
output = I.copy()

gray_image = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY)

  
# Blur using 3 * 3 kernel. 
gray_blurred = cv2.blur(gray_image, (3, 3))


cv2.imshow("gray_blurred", gray_blurred)
cv2.waitKey(0)

# detect circles in the image


circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.2, 20, param1 = 70, 
               param2 = 50, minRadius = 25, maxRadius = 60)

# ensure at least some circles were found
if circles is not None:
	# convert the (x, y) coordinates and radius of the circles to integers
	circles = np.round(circles[0, :]).astype("int")
 
	# loop over the (x, y) coordinates and radius of the circles
	for (x, y, r) in circles:
		# draw the circle in the output image, then draw a rectangle
		# corresponding to the center of the circle
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
 
	# show the output image
	cv2.imshow("output", np.hstack([I, output]))
	cv2.waitKey(0)
else:
    print("No circles found")
    cv2.waitKey(0)


##image = cv2.cvtColor(I,cv2.COLOR_BGR2RGB)
##yuv_image = cv2.cvtColor(I,cv2.COLOR_BGR2YUV)
##hsv_image = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
##
##r, g, b = cv2.split(image)
##y, u, v = cv2.split(yuv_image)
##h, s, v2 = cv2.split(hsv_image)
##
##display images side by side with histogram
##plt.subplot(4,3,1)
##plt.title('rgb image')
##plt.imshow(image)
##plt.subplot(4,3,2)
##plt.title('yuv image')
##plt.imshow(yuv_image)
##plt.subplot(4,3,3)
##plt.title('hsv image')
##plt.imshow(hsv_image)
##plt.subplot(4,3,4)
##plt.imshow(r, cmap='gray')
##plt.subplot(4,3,5)
##plt.imshow(y, cmap='gray')
##plt.subplot(4,3,6)
##plt.imshow(h, cmap='gray')
##plt.subplot(4,3,7)
##plt.imshow(g, cmap='gray')
##plt.subplot(4,3,8)
##plt.imshow(u, cmap='gray')
##plt.subplot(4,3,9)
##plt.imshow(s, cmap='gray')
##plt.subplot(4,3,10)
##plt.imshow(b, cmap='gray')
##plt.subplot(4,3,11)
##plt.imshow(v, cmap='gray')
##plt.subplot(4,3,12)
##plt.imshow(v2, cmap='gray')
##plt.show()
##
