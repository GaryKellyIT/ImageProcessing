'''
Authors: Gary Kelly - C16380531, Patrick O'Connor - C16462144, Tega Orogun - C16518763
Task: Image Processing Project: Traffic Light Detection
Method: 1.User starts the program in command line by running command to start application
          and pass an input image or video as an argument like so 'python FinalVersion.py --input "input/GreenLight.jpg"'.
          Argument is taken in using argparse.
        2.Program checks that the users python version is 3 or higher and the filetype matches
          one of the accepted file types declared in the accepted file types lists.
        3.The program checks whether the file is a video or image file and gives error messages
          if filetype is not accepted.
        4.If the filetype is a video the file is opened and passed frame by frame into our traffic
          light detection function which returns the frame with a bounding box around the detected
          light and a circle on the detected colour traffic light whilst also outputting in the terminal
          the colour of the light detected.
        5.If the filetype is an image it repeats the above step and outputs the resulting image.
        6.The traffic light detection function works like this:
            1)Take in image
            2)Threshold for black
            3)Find largest contour
            4)Create bounding box around this contour
            4)Use this bounding box as region of interest
            5)Create 3 masks using this ROI - one for each of the colours of the traffic light
            6)Perform Hough circles transfor on each of these masks and display resulting circles
            7)Output the colour of the circle found
            8)Return image displaying the bounding box and Hough circle
        7.After displaying the image prompt the user for input - either "w" to write image to current
          directory or "other key" to skip this part. (Videos are automatically written to current directory
          as they are working with multiple frames so waiting for user input would force us to run another loop
          through the video and this would be inefficient)
        8.The user is prompted to either press "a" to enter a new file in which to run the program on or to
          press "q" to quit the program. If a user enters any other character the program will close regardless.
        
Refrences: Used an online tutorial on how to work with the argparse package in order to
          take in command line arguments -
          https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/

          Used a function found on stack overflow to resize an image to a given height or
          width whilst keeping the aspect ratio -
          https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv/44659589#44659589

          Teknomo,K. (2017) Video Analysis using OpenCV-Python
          http://people.revoledu.com/kardi/tutorial/Python/

          HevLfreis “TrafficLight-Detector” Online Github Repoistory
          https://github.com/HevLfreis/TrafficLight-Detector/blob/master/src/main.py?fbclid=IwAR0LNP8zjK_nsFaS9FS15r-X5cpZW18ExW8PDOKaSrLdkFP-eETCwx2u9Qc

          De Charette, R. and Nashashibi, F., 2009, October. Traffic light recognition using image processing compared to learning processes.
          In 2009 IEEE/RSJ International Conference on Intelligent Robots and Systems (pp. 333-338). IEEE.

          Kenan Alkiek, May 4 2018 – “Traffic Light Recognition — A Visual Guide”
          https://medium.com/@kenan.r.alkiek/https-medium-com-kenan-r-alkiek-traffic-light-recognition-505d6ab913b1
'''
import sys
import argparse
import os
import cv2
import numpy as np
import math 
import easygui
from matplotlib import pyplot as plt
from os import path

'''
-------------------------------------------
error checking
-------------------------------------------
'''
#check python version is not less than 3
if sys.version_info[0]<3 :
    print("Version of python 3 does not meet minimum standard. Please install Python 3 or higher!")
    exit()

'''
-------------------------------------------
functions
-------------------------------------------
'''
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

def findTrafficLight(img):
    

    if input_type == "image":
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

    ##cv2.imshow('cimg', cimg)
    ##cv2.imshow('newimg', newimg)

    hsv = cv2.cvtColor(newimg, cv2.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0,170,100])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([160,170,100])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([40,140,90])
    upper_green = np.array([90,255,255])
    lower_yellow = np.array([15,150,150])
    upper_yellow = np.array([35,255,255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskg = cv2.inRange(hsv, lower_green, upper_green)
    masky = cv2.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv2.add(mask1, mask2)
    #cv2.imshow('img', img)
    ##cv2.imshow('maskg', maskg)
    ##cv2.imshow('masky', masky)
    #cv2.imshow('maskr', maskr)
    ##cv2.imshow("g", np.hstack([img, maskg]))
    ##cv2.imshow("r", np.hstack([img, maskr]))
    ##cv2.imshow("y", np.hstack([img, masky]))

    redresult = cv2.bitwise_and(newimg, newimg, mask=maskr)
    yellowresult = cv2.bitwise_and(newimg, newimg, mask=masky)
    greenresult = cv2.bitwise_and(newimg, newimg, mask=maskg)



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
            cv2.circle(newimg, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(newimg, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                    
    if g_circles is not None:
        g_circles = np.round(g_circles[0, :]).astype("int")
     
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in g_circles:
            
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(newimg, (x, y), r, (0, 255, 0), 4)
            
                    
    if y_circles is not None:
        y_circles = np.round(y_circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in y_circles:
            
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(newimg, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(newimg, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

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
            
    x,y,w,h=cv2.boundingRect(contours[0])
    cimg[y:y+h,x:x+w] = newimg
    return cimg

'''
-------------------------------------------
Take in arguments
-------------------------------------------
'''                        
#declare accepted filtetypes
accepted_filetype = [".PNG",".png",".JPEG",".jpg",".jpeg",".JPG"]
accepted_video_filetype = [".MP4",".mp4",".AVI",".avi"]


#take in command line arguments
ap = argparse.ArgumentParser()
arg = ap.add_argument("-i", "--input", required=True,
        help="path to input image/video")
args = vars(ap.parse_args())

input_file_location = "Input/"
output_file_location = "Output/"
'''
-------------------------------------------
Program loop
-------------------------------------------
'''
while True:
    filetype_error = True 
    count = 0
    file_input_name = args["input"]
    args["input"] = input_file_location + args["input"]
    
    #loop through accepted filetypes and ensure that the inputted file ends with one of these
    if path.exists(args["input"]):
        for filetype in range(len(accepted_filetype)) :
            if args["input"].endswith(accepted_filetype[filetype]):
                filetype_error = False
                input_type = "image"
                filetype_used = accepted_filetype[filetype]
                break
            
        for filetype in range(len(accepted_video_filetype)) :
            if args["input"].endswith(accepted_video_filetype[filetype]):
                filetype_error = False
                input_type = "video"
                filetype_used = accepted_video_filetype[filetype]
                break
                
        if filetype_error == True:
            print("Filetype not supported (png/jpg/mp4/avi only)")
            exit()
    else:
        print("File doesn't exist")
        exit()



    fileName=args["input"]


    if input_type == "video":
        cap = cv2.VideoCapture(fileName)          # load the video

        # Default resolutions of the frame are obtained.The default resolutions are system dependent.
        # We convert the resolutions from float to integer.
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
         
        # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        out = cv2.VideoWriter('Output/outpy2.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))


        while(cap.isOpened()):                    # play the video by reading frame by frame
            ret, frame = cap.read()
            if ret==True:
                img = frame
                cimg = findTrafficLight(img)
                cv2.imshow('detected results', cimg)
                out.write(cimg)

            
                cv2.imshow('frame',frame)              # show the video
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # When everything done, release the video capture and video write objects
        cap.release()
        out.release()
    else:
        img = cv2.imread(fileName)
        cimg = findTrafficLight(img)
        cv2.imshow('detected results', cimg)

        #Format file output name based on input filename
        tmpName = file_input_name.split(filetype_used)
        tmpName[len(tmpName)-2] += "output"
        fileOutputName = output_file_location + tmpName[len(tmpName)-2] + filetype_used

        print("Press w to write output to current directory or space to continue")
        key = cv2.waitKey(0)
        #if the 'w' key is pressed, write the resulting image:
        if key == ord("w") and count == 0:
            if input_type == "image":
                cv2.imwrite(fileOutputName,cimg)
                count = 1
                print("Writing image to current directory")
                del key
            
        
    
    print("Press a to choose another file for program or q to quit program")
    key = cv2.waitKey(0)
        
    if key == ord("a"):
        args["input"] = input("Choose next file for reading\n")
        cv2.destroyAllWindows()
        del key

    # if the 'q' key is pressed, quit:
    elif key == ord("q"):
        print("Closing program")
        break

    else:
        print("That wasn't an option, closing program")
        break


cv2.destroyAllWindows()
'''
Concluding comment - The program works exactly how I had hoped when starting this project.
                     Each of the test images and videos provided give correct feedback and
                     the user interface is very easy to manage and allows the user to run the
                     program without having to run the application each time which is pretty
                     neat. All in all I am happy with the performance of the system and eager
                     to see how it performs in a real life scenario with my final year project.
'''
