#####################################################################
import time
import cv2
import os
import sys
import numpy as np
from multiprocessing import Process
from threading import Thread

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
deletionFIFO = "/tmp/deletion.fifo"

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
    newImg = newImg.reshape(newImg.shape[0], newImg.shape[1],1)
    newImg = newImg.astype('float32')
    newImg /= 255
    return newImg

def queryNeuralNetwork(img1, img2):
    img1 = cv2.resize(img1,(128,256))
    img2 = cv2.resize(img2,(128,256))
    
    img1 = img1.reshape(256,128,1)
    img2 = img2.reshape(256,128,1)

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
        for person in people:
            if not person.isActive():
                currentPerson = person.getIdentifier()
                previous = person.getPrevious()
                for previousFrame in previous:
                    prediction = queryNeuralNetwork(personROI,previousFrame)                
                    if prediction < closest:
                        closest = prediction
                        closestPerson = currentPerson
        if closest < 0.5:
            person = people[closestPerson]
            person.addPrevious(personROI)
            return person.getIdentifier()
        else:
            newPerson = Person(len(people))
            newPerson.addPrevious(personROI)
            people.append(newPerson)    
            return newPerson.getIdentifier()

def write(person):
    fifo = open(intsFIFO, "w")
    fifo.write(str(person)+"\n")
    fifo.close()

def processImage():
    while True:
        fifo = open(imagesFIFO, "r")
        data = fifo.read()

        temp = data.split("/")
        if len(temp) > 1:
            img = cv2.imread(data)
            person = whichPerson(img)
            write(person)
            os.remove(data)
        else:
            idToDelete = int(data)
            people[idToDelete].makeInactive()
        fifo.close()

# def processDeletion():
#     while True:
#         fifo = open(deletionFIFO, "r")
#         data = fifo.read()
        
#         idToDelete = int(data)

#         print(len(people))

#         print(idToDelete)

#         people[idToDelete].makeInactive()

#         fifo.close()


model = load_model("saved_models/openWorld.h5", custom_objects={'contrastive_loss': contrastive_loss, 'calc_accuracy': calc_accuracy, 'euclidean_distance': euclidean_distance, 'eucl_dist_output_shape': eucl_dist_output_shape})



if __name__ == '__main__':
    try:
        os.mkfifo(imagesFIFO)
    except:
        print("image fifo exists")

    try:
        os.mkfifo(intsFIFO)
    except:
        print("ints fifo exists")

    try:
        os.mkfifo(deletionFIFO)
    except:
        print("deletion fifo exists")

    try:
        os.mkdir("/tmp/imgs")
    except:
        print("directory exists")

    # p1 = Thread(target=processImage)
    # p1.start()
    # p2 = Thread(target=processDeletion)
    # p2.start()

    processImage()