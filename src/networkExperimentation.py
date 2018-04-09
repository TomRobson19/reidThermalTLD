from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1)  # for reproducibility

import random
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Activation, Flatten, Reshape, Permute
from keras.optimizers import RMSprop
from keras import backend as K
import os
import cv2

import tensorflow as tf
import sklearn
from sklearn import metrics
#import scikitplot as skplt

import matplotlib.pyplot as plt
plt.switch_backend("Agg")

outputFolder = "output"
import time
ts = time.time()
outputFolder = outputFolder+"/"+str(ts).split(".")[0]
tbCallBack = TensorBoard(log_dir=outputFolder+'/log', histogram_freq=0,  write_graph=True, write_images=True)

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

def create_base_network(input_shape):
    #Base network to be shared (eq. to feature extraction)

    input_main = Input(shape=input_shape, dtype='float32')
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_main)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(5, 5))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (7, 7), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
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
x_test = []
y_train = []
y_test = []

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
            if np.random.random() < 0.8:
                x_train.append(img)
                y_train.append(int(target))
                x_train.append(aug)
                y_train.append(int(target))
            else:
                x_test.append(img)
                y_test.append(int(target))
                x_test.append(aug)
                y_test.append(int(target))

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

input_shape = (256, 128, 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255  

num_epochs = 100

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(9)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(9)]
te_pairs, te_y = create_pairs(x_test, digit_indices)

# network definition
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)


# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)

model.compile(loss=contrastive_loss, optimizer='adadelta', metrics=[calc_accuracy])

model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=128,
          epochs=num_epochs,
          verbose=2,
          callbacks=[tbCallBack])

# compute final accuracy on training and test sets
pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, pred)

np.savetxt("tr_predictions.csv",pred, delimiter=",")

pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, pred)

np.savetxt("te_predictions.csv",pred, delimiter=",")

# probs = np.equal(pred.ravel() < 0.5, te_y)

# fpr,tpr,_ = sklearn.metrics.roc_curve(te_y,probs)
# roc_auc = sklearn.metrics.auc(fpr, tpr)

# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.savefig("ROC.png")

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'openWorld.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)



vectorTest = Model(inputs=[input_a, input_b], outputs=[processed_a,processed_b])

test = vectorTest.predict([te_pairs[:, 0], te_pairs[:, 1]])


np.savetxt("vectors1.csv",test[0], delimiter=",")
np.savetxt("vectors2.csv",test[1], delimiter=",")