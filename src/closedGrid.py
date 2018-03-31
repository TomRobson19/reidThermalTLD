# from __future__ import absolute_import
# from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
import keras
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
    num_classes = 9


    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test

def kerasClassifier(x_train, y_train, x_test, y_test):
    input_shape = (256, 128, 1)
    num_classes = 9

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
    final = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_main, outputs=final)

    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=50, verbose=2)

    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)

    K.clear_session()

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=kerasClassifier,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          eval_space=True,
                                          trials=Trials())
print(best_run)

X_train, Y_train, X_test, Y_test = data()
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print("Best performing model chosen hyper-parameters:")
print(best_run)