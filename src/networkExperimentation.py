from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import random
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Lambda, Conv2D, MaxPooling2D, Activation, Flatten, ZeroPadding2D
from keras.optimizers import RMSprop
from keras import backend as K
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import cv2



from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2




outputFolder = "output"
import time
ts = time.time()
outputFolder = outputFolder+"/"+str(ts).split(".")[0]

tbCallBack = TensorBoard(log_dir=outputFolder+'/log', histogram_freq=0,  write_graph=True, write_images=True)

def euclidean_distance(vects):
    x, y = vects
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
    n = min([len(digit_indices[d]) for d in range(8)]) - 1
    for d in range(8):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, 8)
            dn = (d + inc) % 8
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


# generator version - currently yields one positive and one negative pair per call

def generate_pairs(x, digit_indices, batch_size):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    current = 0
    pairs1 = np.zeros((batch_size, 256, 128, 1))
    pairs2 = np.zeros((batch_size, 256, 128, 1))
    labels = np.zeros((batch_size, 2))

    next_pair = True
    #n = min([len(digit_indices[d]) for d in range(8)]) - 1
    while(True):
        for d in range(8):
            n = len(digit_indices[d])-1
            for i in range(n):
                if next_pair == True:
                    z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                    pairs1[current] = x[z1]
                    pairs2[current] = x[z2]
                    labels[current] = [1, 0]
                    next_pair = False
                else:
                    inc = random.randrange(1, 8)
                    dn = (d + inc) % 8
                    j = random.randint(0,len(digit_indices[dn])-1)
                    z1, z2 = digit_indices[d][i], digit_indices[dn][j]
                    pairs1[current] = x[z1]
                    pairs2[current] = x[z2]
                    labels[current] = [0, 1]
                    next_pair = True 
                current += 1

                if current == batch_size:
                    yield [pairs1, pairs2], labels
                    pairs1 = np.zeros((batch_size, 256, 128, 1))
                    pairs2 = np.zeros((batch_size, 256, 128, 1))
                    labels = np.zeros((batch_size, 2))
                    current = 0

def create_base_network(input_shape):
    #Base network to be shared (eq. to feature extraction).
    
    # nb_filters = 128
    # pool_size = (8,8)
    # kernel_size = (8,8)
    # model = Sequential()
    # model.add(Conv2D(nb_filters, kernel_size, padding='same', input_shape=input_shape, activation='relu'))
    # model.add(Conv2D(nb_filters, kernel_size, activation='relu'))
    # model.add(Conv2D(nb_filters, kernel_size, activation='relu'))
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(128, activation='relu'))




    # model = Sequential()
    # model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(128, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(256, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(Conv2D(512, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(4096, activation='relu'))




    # img_input = Input(input_shape)
    # x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
    # x = BatchNormalization(name='block1_conv1_bn')(x)
    # x = Activation('relu', name='block1_conv1_act')(x)
    # x = Conv2D(64, (3, 3), use_bias=False)(x)
    # x = BatchNormalization(name='block1_conv2_bn')(x)
    # x = Activation('relu', name='block1_conv2_act')(x)

    # residual = Conv2D(128, (1, 1), strides=(2, 2),
    #                   padding='same', use_bias=False)(x)
    # residual = BatchNormalization()(residual)

    # x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    # x = BatchNormalization(name='block2_sepconv1_bn')(x)
    # x = Activation('relu', name='block2_sepconv2_act')(x)
    # x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
    # x = BatchNormalization(name='block2_sepconv2_bn')(x)

    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # x = layers.add([x, residual])

    # residual = Conv2D(256, (1, 1), strides=(2, 2),
    #                   padding='same', use_bias=False)(x)
    # residual = BatchNormalization()(residual)

    # x = Activation('relu', name='block3_sepconv1_act')(x)
    # x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    # x = BatchNormalization(name='block3_sepconv1_bn')(x)
    # x = Activation('relu', name='block3_sepconv2_act')(x)
    # x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
    # x = BatchNormalization(name='block3_sepconv2_bn')(x)

    # x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    # x = layers.add([x, residual])
    # x = GlobalAveragePooling2D()(x)

    # model = Model(img_input, x)

    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=16, kernel_size=(7, 7), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(.5))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=256, kernel_size=(3, 3), padding='same'))
    model.add(GlobalAveragePooling2D())


    return model

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()

X_train = []
X_test = []
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
        if int(target) < 8:
            img = cv2.imread(os.path.join(image_dir, target, img_file))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            if np.random.random() < 0.8:
                X_train.append(img)
                y_train.append(int(target))
            else:
                X_test.append(img)
                y_test.append(int(target))

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

input_shape = (256, 128, 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

BATCH_SIZE = 8
num_epochs = 4

digit_indices = [np.where(y_train == i)[0] for i in range(8)]
train_generator = generate_pairs(X_train, digit_indices, BATCH_SIZE)

digit_indices = [np.where(y_test == i)[0] for i in range(8)]
test_generator = generate_pairs(X_test, digit_indices, BATCH_SIZE)

# GENERATOR TESTING
# while True:
#     test = random.randint(0,7)
#     [X1,X2],Y = next(train_generator)
#     cv2.imshow("test1", X1[test])
#     cv2.imshow("test2", X2[test])
#     print(Y[test])
#     cv2.waitKey(10000)
# counter = 0
# while True:
#     print(next(train_generator))
#     print(counter)
#     counter += 1

# network definition
# base_network = create_base_network(input_dim, X_train)
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

# train - try with Adam and Adadelta
#rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer="adadelta", metrics=["accuracy"])

num_train_steps = len(X_train) // BATCH_SIZE
num_val_steps = len(X_test) // BATCH_SIZE

BEST_MODEL_FILE = os.path.join("test", "checkpoint.h5")

checkpoint = ModelCheckpoint(filepath=BEST_MODEL_FILE, save_best_only=True)
history = model.fit_generator(train_generator, 
                             steps_per_epoch=num_train_steps-2,
                             epochs=num_epochs,
                             validation_data=test_generator,
                             validation_steps=num_val_steps-2, 
                             verbose=2,
                             callbacks=[tbCallBack,checkpoint])

FINAL_MODEL_FILE = os.path.join("test", "model.h5")
model.save(FINAL_MODEL_FILE, overwrite=True)

def evaluate_model(model):
    ytest, ytest_ = [], []
    num_test_steps = len(X_test) // BATCH_SIZE
    curr_test_steps = 0
    for [X1test, X2test], Ytest in test_generator:
        if curr_test_steps > num_test_steps:
            break
        Ytest_ = model.predict([X1test, X2test])
        ytest.extend(np.argmax(Ytest, axis=1).tolist())
        ytest_.extend(np.argmax(Ytest_, axis=1).tolist())
        curr_test_steps += 1
    acc = accuracy_score(ytest, ytest_)
    cm = confusion_matrix(ytest, ytest_)
    return acc, cm

print("==== Evaluation Results: final model on test set ====")
final_model = load_model(FINAL_MODEL_FILE,custom_objects={'contrastive_loss': contrastive_loss})
acc, cm = evaluate_model(final_model)
print("Accuracy Score: {:.3f}".format(acc))
print("Confusion Matrix")
print(cm)