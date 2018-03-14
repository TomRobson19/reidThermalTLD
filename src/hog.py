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
import numpy as np
from person import Person

from keras.models import load_model, Model
from keras import backend as K

import tensorflow as tf

#####################################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

EVENT_LOOP_DELAY = 40	# delay for GUI window
                        # 40 ms equates to 1000ms/25fps = 40ms per frame
#####################################################################

def euclidean_distance(vects):
    x, y = vects

    #return K.cast(K.less(K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True)),0.5),"float32")

    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def calc_accuracy(labels, predictions):
    '''accuracy function for compilation'''
    return K.mean(K.equal(labels, K.cast(K.less(predictions,0.5),"float32")))


def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

def draw_detections(img, x,y,w,h, colour, thickness = 1):
    # the HOG detector returns slightly larger rectangles than the real objects.
    # so we slightly shrink the rectangles to get a nicer output.
    pad_w, pad_h = int(0.15*w), int(0.05*h)
    cv2.rectangle(img, (x, y), (x+w, y+h), thickness)

def convertForKeras(img):
    newImg = np.array(img)
    newImg = newImg.reshape(newImg.shape[0], newImg.shape[1],1)
    newImg = newImg.astype('float32')
    newImg /= 255
    return newImg

def queryNeuralNetwork(img1, img2):
    img1 = cv2.resize(img1,(128,256))
    img2 = cv2.resize(img2,(128,256))
    
    # cv2.imshow("test1",img1)
    # cv2.imshow("test2",img2)

    # cv2.waitKey(10000)

    img1 = img1.reshape(256,128,1)
    img2 = img2.reshape(256,128,1)

    img1 = np.array([img1])
    img2 = np.array([img2])

    prediction = model.predict([img1,img2])
    return prediction[0][0]


#####################################################################
model = load_model("saved_models/openWorld.h5", custom_objects={'contrastive_loss': contrastive_loss, 'calc_accuracy': calc_accuracy, 'euclidean_distance': euclidean_distance, 'eucl_dist_output_shape': eucl_dist_output_shape})

padding = 40
width = 100
height = 100

people = []

# define video capture object

cap = cv2.VideoCapture()

# define display window name

windowName = "HOG pedestrain detection" # window name

# if command line arguments are provided try to read video_name
# otherwise default to capture from attached H/W camera

def runOnSingleCamera(video_file):

    cap.open(video_file)

    # create window by name (as resizable)

    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 640,480)

    # set up HoG detector

    mog = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=16, detectShadows=True)

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )

    keep_processing = True

    while (keep_processing):

        ret, img = cap.read()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        displayImage = img

        fgmask = mog.apply(img)

        fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        fgdilated = cv2.dilate(fgthres, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)

        _, contours, hierarchy = cv2.findContours(fgthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            
            x,y,w,h = cv2.boundingRect(i)

            originalx = x
            originaly = y

            x = int(max(0, x - padding / 100.0 * w))
            y = int(max(0, y - padding / 100.0 * h))

            w = int(min(img.shape[1] - 1, (w + 2 * padding / 100.0 * w)))
            h = int(min(img.shape[0] - 1, (h + 2 * padding / 100.0 * h)))

            if ((w >= width) and (h >= height) and (x + w < img.shape[1]) and (y + h < img.shape[0])):

                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                roi = img[y:h+y,x:w+x]


                #perform HOG based pedestrain detection

                found, w = hog.detectMultiScale(roi, winStride=(8,8), padding=(32,32), scale=1.05)
                found_filtered = []

                for ri, r in enumerate(found):
                    for qi, q in enumerate(found):
                        if ri != qi and inside(r, q):
                            break
                        else:
                            found_filtered.append(r)

                #draw_detections(img, found)
                # draw_detections(displayImage, found_filtered, colour, 3)

                for x, y, w, h in found_filtered:
                    #w, h = int(0.15*w), int(0.05*h)
                    personROI = roi[y:h+y,x:w+x]

                    personROI = convertForKeras(personROI)
                    if len(people) == 0:
                        newPerson = Person(0)
                        newPerson.addPrevious(personROI)
                        people.append(newPerson)
                        draw_detections(displayImage, originalx, originaly, w,h, newPerson.getColour(), 3)
                    else:
                        closest = 100
                        closestPerson = 100
                        for person in people:
                            currentPerson = person.getIdentifier()
                            previous = person.getPrevious()
                            for previousFrame in previous:
                                prediction = queryNeuralNetwork(personROI,previousFrame)

                                print(currentPerson, prediction)
                                
                                if prediction < closest:
                                    closest = prediction
                                    closestPerson = currentPerson
                        if closest < 0.5:
                            person = people[closestPerson]
                            person.addPrevious(personROI)
                            draw_detections(displayImage, originalx, originaly, w, h, person.getColour(), 3)
                            print("REID")
                        else:
                            newPerson = Person(len(people))
                            newPerson.addPrevious(personROI)
                            people.append(newPerson)
                            draw_detections(displayImage, originalx, originaly, w, h, newPerson.getColour(), 3)
                            print("NEW")
        # display image

        cv2.imshow(windowName,displayImage)

        # if user presses "x" then exit

        key = cv2.waitKey(1) & 0xFF # wait 200ms (i.e. 1000ms / 5 fps = 200 ms)
        if (key == ord('x')):
            keep_processing = False

    # close all windows

    cv2.destroyAllWindows()

if __name__ == '__main__':
    runOnSingleCamera("data/Dataset1/betaInput.webm")