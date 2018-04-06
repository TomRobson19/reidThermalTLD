from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1)  # for reproducibility

import random
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Activation, Flatten, Reshape, Permute
from keras.optimizers import RMSprop
from keras import backend as K
import os
import cv2

import tensorflow as tf
import sklearn
from sklearn import metrics
#import scikitplot as skplt

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


def create_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(5)]) - 1
    for d in range(5):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 5)
            dn = (d + inc) % 5
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_base_network(input_shape):
    #Base network to be shared (eq. to feature extraction)

    input_main = Input(shape=input_shape, dtype='float32')
    x = Conv2D(32, (3, 3), padding='same', activation='tanh')(input_main)
    x = Conv2D(16, (5, 5), activation='tanh')(x)
    x = MaxPooling2D(pool_size=(5,5))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (7,7), activation='tanh')(x)
    x = MaxPooling2D(pool_size=(3,3))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    #x = Dropout(0.25)(x)
    x = Dense(16, activation='relu')(x)

    model = Model(inputs=input_main, outputs=x)
    return model


def calc_accuracy(labels, predictions):
    '''accuracy function for compilation'''
    return K.mean(K.equal(labels, K.cast(K.less(predictions,0.5),"float32")))

def compute_accuracy(labels, predictions): 
    '''final computation of accuracy'''
    #return labels[predictions.ravel() < 0.5].mean()
    return np.mean(np.equal(predictions.ravel() < 0.5, labels))

# the data, shuffled and split between train and test sets
x_train = []
y_train = []

image_dir = "people"
img_groups = {}
for person in os.listdir(image_dir): 
    for img_file in os.listdir(image_dir + "/" + person):
        if person in img_groups:
            img_groups[person].append(img_file)
        else:
            img_groups[person]=[img_file]

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


for target in img_groups:
    for img_file in img_groups[target]:
        if int(target) < 5:
            img = cv2.imread(os.path.join(image_dir, target, img_file))
            aug = datagen.random_transform(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            aug = cv2.cvtColor(aug, cv2.COLOR_RGB2GRAY)

            x_train.append(img)
            y_train.append(int(target))
            x_train.append(aug)
            y_train.append(int(target))

x_train = np.array(x_train)
y_train = np.array(y_train)

input_shape = (256, 128, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train.astype('float32')
x_train /= 255

num_epochs = 100

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(5)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

s = np.arange(tr_pairs.shape[0]) 
np.random.shuffle(s) 
 
tr_pairs = tr_pairs[s] 
tr_y = tr_y[s] 

model = load_model("saved_models/openWorld.h5", custom_objects={'contrastive_loss': contrastive_loss, 'calc_accuracy': calc_accuracy, 'euclidean_distance': euclidean_distance, 'eucl_dist_output_shape': eucl_dist_output_shape})

print("predicting")

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, pred)

np.savetxt("tr_predictions.csv",pred, delimiter=",")
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))