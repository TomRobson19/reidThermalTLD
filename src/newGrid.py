# from __future__ import absolute_import
# from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

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

from hyperas.utils import eval_hyperopt_space

import tensorflow as tf
import sklearn
#import scikitplot as skplt


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

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

def calc_accuracy(labels, predictions):
    '''accuracy function for compilation'''
    return K.mean(K.equal(labels, K.cast(K.less(predictions,0.5),"float32")))

def compute_accuracy(labels, predictions): 
    '''final computation of accuracy'''
    #return labels[predictions.ravel() < 0.5].mean()
    return np.mean(np.equal(predictions.ravel() < 0.5, labels))

# network definition
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from hyperopt import Trials, STATUS_OK, tpe

def data():

    x_train = []
    x_val = []
    x_test = []
    y_train = []
    y_val = []
    y_test = []

    image_dir = "newPeople"
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
            rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
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

                rand = np.random.random()
                if rand < 0.6:
                    x_train.append(img)
                    y_train.append(int(target))
                    x_train.append(aug)
                    y_train.append(int(target))
                elif rand > 0.6 and rand < 0.8:
                    x_val.append(img)
                    y_val.append(int(target))
                    x_val.append(aug)
                    y_val.append(int(target))
                else:
                    x_test.append(img)
                    y_test.append(int(target))
                    x_test.append(aug)
                    y_test.append(int(target))

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    input_shape = (256, 128, 1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_val /= 255  
    x_test /= 255

    num_epochs = 50

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_train == i)[0] for i in range(5)]
    tr_pairs, tr_y = create_pairs(x_train, digit_indices)

    digit_indices = [np.where(y_val == i)[0] for i in range(5)]
    val_pairs, val_y = create_pairs(x_val, digit_indices)

    digit_indices = [np.where(y_test == i)[0] for i in range(5)]
    te_pairs, te_y = create_pairs(x_test, digit_indices)

    x_train = [tr_pairs[:, 0], tr_pairs[:, 1]]
    y_train = tr_y
    x_test = [val_pairs[:, 0], val_pairs[:, 1]]
    y_test = val_y

    x_eval = [te_pairs[:, 0], te_pairs[:, 1]]
    y_eval = te_y

    return x_train, y_train, x_test, y_test, x_eval, y_eval

def kerasClassifier(x_train, y_train, x_test, y_test, x_eval, y_eval):
    input_shape = (256, 128, 1)

    input_main = Input(shape=input_shape, dtype='float32')
    x = Conv2D({{choice([8,16,32,64])}}, {{choice([1,2,3,4,5,6,7,8,9,10])}}, padding='same', activation={{choice(['relu', 'tanh'])}})(input_main)
    x = Conv2D({{choice([8,16,32,64])}}, {{choice([1,2,3,4,5,6,7,8,9,10])}}, padding='same', activation={{choice(['relu', 'tanh'])}})(x)
    x = MaxPooling2D(pool_size={{choice([1,2,3,4,5,6,7,8,9,10])}})(x)
    x = Dropout({{choice([0,0,0.1,0.2,0.3,0.4,0.5])}})(x) 
   
    x = Conv2D({{choice([8,16,32,64])}}, {{choice([1,2,3,4,5,6,7,8,9,10])}}, padding='same', activation={{choice(['relu', 'tanh'])}})(x)
    x = Conv2D({{choice([8,16,32,64])}}, {{choice([1,2,3,4,5,6,7,8,9,10])}}, padding='same', activation={{choice(['relu', 'tanh'])}})(x)
    x = MaxPooling2D(pool_size={{choice([1,2,3,4,5,6,7,8,9,10])}})(x)
    x = Dropout({{choice([0,0,0.1,0.2,0.3,0.4,0.5])}})(x) 
    
    x = Flatten()(x)
    x = Dropout({{choice([0,0,0.1,0.2,0.3,0.4,0.5])}})(x) 
    x = Dense({{choice([16,32,64,128, 256])}}, activation={{choice(['relu', 'tanh'])}})(x)

    base_network = Model(inputs=input_main, outputs=x)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])


    model = Model(inputs=[input_a, input_b], outputs=distance)
    model.compile(loss=contrastive_loss, optimizer='adadelta', metrics=[calc_accuracy])

    model.fit(x_train,y_train,validation_data=(x_test,y_test),batch_size=128,epochs=num_epochs, verbose=2)

    score, acc = model.evaluate(x_eval, y_eval, verbose=0)
    print('Test accuracy:', acc)

    K.clear_session()

    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=kerasClassifier,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=25,
                                          functions=[euclidean_distance,eucl_dist_output_shape, contrastive_loss, calc_accuracy, create_pairs],
                                          eval_space=True,
                                          trials=Trials())
print(best_run)

# X_train, Y_train, X_test, Y_test = data()
# print("Evalutation of best performing model:")
# print(best_model.evaluate(X_test, Y_test))
# print("Best performing model chosen hyper-parameters:")
# print(best_run)