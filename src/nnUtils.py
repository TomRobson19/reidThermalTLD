#####################################################################
import time
import cv2
import os
import sys
import numpy as np
from multiprocessing import Process
from threading import Thread
import shutil

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
imagesFIFO = "/tmp/images.fifo"
intsFIFO = "/tmp/ints.fifo"

people = []

#####################################################################

def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def calc_accuracy(labels, predictions):
    return K.mean(K.equal(labels, K.cast(K.less(predictions,0.5),"float32")))

#####################################################################

def convertForKeras(img):
    newImg = np.array(img)
    newImg = newImg.reshape(256, 128,1)
    newImg = newImg.astype('float32')
    newImg /= 255
    return newImg

def queryNeuralNetwork(img1, img2):
    img1 = np.array([img1])
    img2 = np.array([img2])

    prediction = model.predict([img1,img2])

    return prediction[0][0]

def whichPerson(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    personROI = convertForKeras(img)

    if len(people) == 0:
        newPerson = Person(0)
        newPerson.addPrevious(personROI)
        people.append(newPerson)
        return newPerson.getIdentifier()
    else:
        closest = 100
        closestPerson = 100
        closestFrame = None
        save = False
        for person in people:
            print(str(person.getIdentifier()) + " activity is " + str(person.isActive()))
            if not person.isActive():
                save=True
                currentPerson = person.getIdentifier()
                previous = person.getPrevious()
                for previousFrame in previous:
                    prediction = queryNeuralNetwork(personROI,previousFrame)
                    print("person "+str(person.getIdentifier())+" distance = "+str(prediction))                
                    if prediction < closest:
                        closest = prediction
                        closestPerson = currentPerson
                        closestFrame = previousFrame
        if save:

            concat = np.concatenate((personROI, closestFrame), axis=1)
            concat *= 255
            cv2.imwrite("classificationsREID/"+str(closest)+" "+str(closestPerson)+".jpg",concat)

        # count the number of times these are different people and <0.5, or the same person and >0.5

        if closest < 0.5:
            person = people[closestPerson]
            person.addPrevious(personROI)
            return person.getIdentifier()
        else:
            nextPerson = Person(len(people))
            nextPerson.addPrevious(personROI)
            people.append(nextPerson)
            return nextPerson.getIdentifier()

def write(person):
    fifo = open(intsFIFO, "w")
    fifo.write(str(person)+"\n")
    fifo.close()

def processImage():
    fifo = open(imagesFIFO, "r")
    while True:
        lines = fifo.readlines()

        for data in lines:
            temp = data.split("/")
            if data == '':
                continue
            elif len(temp) > 1:
                img = cv2.imread(data)
                person = whichPerson(img)
                print("classified as " + str(person))
                write(person)
                os.remove(data)
            else:
                print("delete " + data)
                idToDelete = int(data)
                people[idToDelete].makeInactive()
    fifo.close()

model = load_model("saved_models/openWorld.h5", custom_objects={'contrastive_loss': contrastive_loss, 'calc_accuracy': calc_accuracy, 'euclidean_distance': euclidean_distance, 'eucl_dist_output_shape': eucl_dist_output_shape})



if __name__ == '__main__':

    try:
        os.remove(imagesFIFO)
        os.mkfifo(imagesFIFO)
    except:
        print("making new imagesFIFO")
        os.mkfifo(imagesFIFO)

    try:
        os.remove(intsFIFO)
        os.mkfifo(intsFIFO)
    except:
        print("making new intsFIFO")
        os.mkfifo(intsFIFO)

    try:
        shutil.rmtree("/tmp/imgs")
        os.mkdir("/tmp/imgs")
    except:
        print("making new directory")
        os.mkdir("/tmp/imgs")

    print("processing")
    processImage()