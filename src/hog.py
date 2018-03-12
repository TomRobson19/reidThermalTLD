#####################################################################

# Example : HOG pedestrain detection from a video file
# specified on the command line (e.g. FILE.py video_file) or from an
# attached web camera

# Author : Toby Breckon, toby.breckon@durham.ac.uk

# Copyright (c) 2015 / 2016 School of Engineering & Computing Science,
#                    Durham University, UK
# License : LGPL - http://www.gnu.org/licenses/lgpl.html

#####################################################################

import cv2
import sys

#####################################################################

keep_processing = True;
camera_to_use = 1;
EVENT_LOOP_DELAY = 40;	# delay for GUI window
                        # 40 ms equates to 1000ms/25fps = 40ms per frame

#####################################################################

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 255), thickness)

#####################################################################

padding = 40
width = 100
height = 100

# define video capture object

cap = cv2.VideoCapture();

# define display window name

windowName = "HOG pedestrain detection"; # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

if (cap.open("/home/tom/Documents/fourthYearWork/reidThermalTLD/src/data/Dataset1/betaInput.webm")):

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL);
    cv2.resizeWindow(windowName, 640,480)

    # set up HoG detector

    mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True);

    hog = cv2.HOGDescriptor();
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() );

    while (keep_processing):

        ret, img = cap.read();

        fgmask = mog.apply(img)

        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1];
        fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3);

        _, contours, hierarchy = cv2.findContours(fgthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

        for i in contours:
            
            x,y,w,h = cv2.boundingRect(i)

            x = int(max(0, x - padding / 100.0 * w))
            y = int(max(0, y - padding / 100.0 * h))

            w = int(min(img.shape[1] - 1, (w + 2 * padding / 100.0 * w)))
            h = int(min(img.shape[0] - 1, (h + 2 * padding / 100.0 * h)))

            if ((w >= width) and (h >= height) and (x + w < img.shape[1]) and (y + h < img.shape[0])):

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                roi = img[y:h+y,x:w+x]


                #perform HOG based pedestrain detection

                found, w = hog.detectMultiScale(img, winStride=(8,8), padding=(32,32), scale=1.05)
                found_filtered = []

                for ri, r in enumerate(found):
                    for qi, q in enumerate(found):
                        if ri != qi and inside(r, q):
                            break
                        else:
                            found_filtered.append(r)

                draw_detections(img, found)
                draw_detections(img, found_filtered, 3)

        # display image

        cv2.imshow(windowName,img);

        # if user presses "x" then exit

        key = cv2.waitKey(1) & 0xFF; # wait 200ms (i.e. 1000ms / 5 fps = 200 ms)
        if (key == ord('x')):
            keep_processing = False;

    # close all windows

    cv2.destroyAllWindows()