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

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

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
    n = min([len(digit_indices[d]) for d in range(9)]) - 1
    for d in range(9):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 9)
            dn = (d + inc) % 9
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

# def create_base_network(input_shape):
#     #Base network to be shared (eq. to feature extraction)

#     input_main = Input(shape=input_shape, dtype='float32')
#     x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_main)
#     x = Conv2D(16, (5, 5), activation='relu')(x)
#     x = MaxPooling2D(pool_size=(5,5))(x)
#     x = Dropout(0.25)(x)

#     x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
#     x = Conv2D(32, (7,7), activation='relu')(x)
#     x = MaxPooling2D(pool_size=(3,3))(x)
#     x = Dropout(0.25)(x)

#     x = Flatten()(x)
#     #x = Dropout(0.25)(x)
#     x = Dense(16, activation='relu')(x)

#     model = Model(inputs=input_main, outputs=x)
#     return model


def calc_accuracy(labels, predictions):
    '''accuracy function for compilation'''
    return K.mean(K.equal(labels, K.cast(K.less(predictions,0.5),"float32")))

def compute_accuracy(labels, predictions): 
    '''final computation of accuracy'''
    #return labels[predictions.ravel() < 0.5].mean()
    return np.mean(np.equal(predictions.ravel() < 0.5, labels))

# the data, shuffled and split between train and test sets
x_eval = []
y_eval = []

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
        if int(target) < 9:
            img = cv2.imread(os.path.join(image_dir, target, img_file))
            aug = datagen.random_transform(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            aug = cv2.cvtColor(aug, cv2.COLOR_RGB2GRAY)

            x_eval.append(img)
            y_eval.append(int(target))
            x_eval.append(aug)
            y_eval.append(int(target))

x_eval = np.array(x_eval)
y_eval = np.array(y_eval)

input_shape = (256, 128, 1)
x_eval = x_eval.reshape(x_eval.shape[0], x_eval.shape[1], x_eval.shape[2], 1)
x_eval = x_eval.astype('float32')
x_eval /= 255

# y_eval = y_eval[]
# x_eval = x_eval[]

# create training+test positive and negative pairs
digit_indices = [np.where(y_eval == i)[0] for i in range(9)]
ev_pairs, ev_y = create_pairs(x_eval, digit_indices)

# s = np.arange(ev_pairs.shape[0]) 
# np.random.shuffle(s) 
 
# ev_pairs = ev_pairs[s] 
# ev_y = ev_y[s] 

model = load_model("saved_models/openWorld.h5", custom_objects={'contrastive_loss': contrastive_loss, 'calc_accuracy': calc_accuracy, 'euclidean_distance': euclidean_distance, 'eucl_dist_output_shape': eucl_dist_output_shape})

print("predicting")

#compute final accuracy on training and test sets
pred = model.predict([ev_pairs[:, 0], ev_pairs[:, 1]])
np.savetxt("other_data_predictions.csv",pred, delimiter=",")
# print(pred)
ev_acc = compute_accuracy(ev_y, pred)

# np.savetxt("eval_predictions.csv",pred, delimiter=",")
print('* Accuracy on eval set: %0.2f%%' % (100 * ev_acc))

# def queryNeuralNetwork(img1, img2):
#     concat = np.concatenate((img1, img2), axis=1)
#     concat *= 255

#     img1 = np.array([img1])
#     img2 = np.array([img2])

#     prediction = model.predict([img1,img2])

#     if(prediction[0][0] < 0.5):
#         cv2.imwrite("classificationsCNN/positiveTrainTest/"+str(prediction[0][0])+".jpg",concat)
#     else:
#         cv2.imwrite("classificationsCNN/negativeTrainTest/"+str(prediction[0][0])+".jpg",concat)


#     return prediction[0][0]

# for i in ev_pairs:
#     prediction = queryNeuralNetwork(i[0],i[1])