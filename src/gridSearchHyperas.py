# from __future__ import absolute_import
# from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Activation, Flatten, Reshape, Permute
from keras.optimizers import RMSprop
from keras import backend as K
import os
import cv2

from hyperas.utils import eval_hyperopt_space

import tensorflow as tf
import sklearn
#import scikitplot as skplt


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform




# import matplotlib.pyplot as plt
# plt.switch_backend("Agg")

# outputFolder = "output"
# import time
# ts = time.time()
# outputFolder = outputFolder+"/"+str(ts).split(".")[0]
# tbCallBack = TensorBoard(log_dir=outputFolder+'/log', histogram_freq=0,  write_graph=True, write_images=True)

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

# def create_base_network(input_shape, activation='relu',conv1=32,conv2=32,conv3=64,conv4=64,pool1=2,pool2=2,size=3,dense=256):
#     #Base network to be shared (eq. to feature extraction)

#     #model = Sequential()
#     input_main = Input(shape=input_shape, dtype='float32')
#     x = Conv2D(conv1, size, padding='same', activation=activation)(input_main)
#     x = Conv2D(conv2, size, padding='same', activation=activation)(x)
#     x = MaxPooling2D(pool_size=pool1)(x)
#     x = Dropout(0.25)(x)

#     x = Conv2D(conv3, size, padding='same', activation=activation)(x)
#     x = Conv2D(conv4, size, padding='same', activation=activation)(x)
#     x = MaxPooling2D(pool_size=pool2)(x)
#     x = Dropout(0.25)(x)

#     x = Flatten()(x)
#     x = Dense(dense, activation=activation)(x)

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
# x_train = []
# x_test = []
# y_train = []
# y_test = []

# image_dir = "people"
# img_groups = {}
# for person in os.listdir(image_dir): 
#     for img_file in os.listdir(image_dir + "/" + person):
#         if person in img_groups:
#             img_groups[person].append(img_file)
#         else:
#             img_groups[person]=[img_file]

# for target in img_groups:
#     for img_file in img_groups[target]:
#         if int(target) < 8:
#             img = cv2.imread(os.path.join(image_dir, target, img_file))
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             if np.random.random() < 0.8:
#                 x_train.append(img)
#                 y_train.append(int(target))
#             else:
#                 x_test.append(img)
#                 y_test.append(int(target))

# x_train = np.array(x_train)
# x_test = np.array(x_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)

# input_shape = (256, 128, 1)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255  

# num_epochs = 50

# # create training+test positive and negative pairs
# digit_indices = [np.where(y_train == i)[0] for i in range(8)]
# tr_pairs, tr_y = create_pairs(x_train, digit_indices)

# digit_indices = [np.where(y_test == i)[0] for i in range(8)]
# te_pairs, te_y = create_pairs(x_test, digit_indices)


# network definition
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

def data():

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

    for target in img_groups:
        for img_file in img_groups[target]:
            if int(target) < 9:
                img = cv2.imread(os.path.join(image_dir, target, img_file))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                if np.random.random() < 0.8:
                    x_train.append(img)
                    y_train.append(int(target))
                else:
                    x_test.append(img)
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

    num_epochs = 50

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(9)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(9)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    x_train = [tr_pairs[:, 0], tr_pairs[:, 1]]
    y_train = tr_y
    x_test = [te_pairs[:, 0], te_pairs[:, 1]]
    y_test = te_y
    return x_train, y_train, x_test, y_test

def kerasClassifier(x_train, y_train, x_test, y_test):
    input_shape = (256, 128, 1)

    input_main = Input(shape=input_shape, dtype='float32')
    x = Conv2D({{choice([16,32,64])}}, {{choice([1,2,3,4,5,6,7,8,9,10])}}, padding='same', activation={{choice(['relu', 'tanh'])}})(input_main)
    x = Conv2D({{choice([16,32,64])}}, {{choice([1,2,3,4,5,6,7,8,9,10])}}, padding='same', activation={{choice(['relu', 'tanh'])}})(x)
    x = MaxPooling2D(pool_size={{choice([1,2,3,4,5,6,7,8,9,10])}})(x)
    x = Dropout(0.25)(x)

    x = Conv2D({{choice([16,32,64])}}, {{choice([1,2,3,4,5,6,7,8,9,10])}}, padding='same', activation={{choice(['relu', 'tanh'])}})(x)
    x = Conv2D({{choice([16,32,64])}}, {{choice([1,2,3,4,5,6,7,8,9,10])}}, padding='same', activation={{choice(['relu', 'tanh'])}})(x)
    x = MaxPooling2D(pool_size={{choice([1,2,3,4,5,6,7,8,9,10])}})(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense({{choice([16,32,64,128, 256])}}, activation={{choice(['relu', 'tanh'])}})(x)

    base_network = Model(inputs=input_main, outputs=x)

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

    model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=num_epochs, verbose=2)

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)

    K.clear_session()

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=kerasClassifier,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          functions=[euclidean_distance,eucl_dist_output_shape, contrastive_loss, calc_accuracy, create_pairs],
                                          eval_space=True,
                                          trials=Trials())
print(best_run)

X_train, Y_train, X_test, Y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)



# model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#           validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
#           batch_size=128,
#           epochs=num_epochs,
#           verbose=2,
#           callbacks=[tbCallBack])

# # compute final accuracy on training and test sets
# pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
# tr_acc = compute_accuracy(tr_y, pred)

# pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
# te_acc = compute_accuracy(te_y, pred)

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

# print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
# print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))


# save_dir = os.path.join(os.getcwd(), 'saved_models')
# model_name = 'openWorld.h5'
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# model_path = os.path.join(save_dir, model_name)
# model.save(model_path)
# print('Saved trained model at %s ' % model_path)



# vectorTest = Model(inputs=[input_a, input_b], outputs=[processed_a,processed_b])

# test = vectorTest.predict([te_pairs[:, 0], te_pairs[:, 1]])
# print(test)

# np.savetxt("vectors.csv",test[0], delimiter=",")