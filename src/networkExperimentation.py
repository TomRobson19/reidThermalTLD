from __future__ import absolute_import
from __future__ import print_function
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

import tensorflow as tf
import sklearn
import scikitplot as skplt

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


def create_eval_pairs(x, digit_indices):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(8,10)]) - 1
    for d in range(8,10):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(8, 9)
            dn = (d + inc) % 2
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

shape = 0
def flatten_bodge(x):
    return tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1]*tf.shape(x)[2]*tf.shape(x)[3]])

def create_base_network(input_shape):
    #Base network to be shared (eq. to feature extraction)

    #model = Sequential()
    input_main = Input(shape=input_shape, dtype='float32')
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_main)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)

    # x = Lambda(flatten_bodge, dtype='float32')(x)

    # a,b,c,d = x.get_shape().as_list()
    # a = b*c*d

    # # x = Permute([1, 2, 3])(x)  # Indicate NHWC data layout
    # x = Reshape((a,))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu', name="final")(x)

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

for target in img_groups:
    for img_file in img_groups[target]:
        if int(target) < 8:
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

num_epochs = 100

# create training+test positive and negative pairs
digit_indices = [np.where(y_train == i)[0] for i in range(8)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices = [np.where(y_test == i)[0] for i in range(8)]
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

pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
te_acc = compute_accuracy(te_y, pred)

probs = np.equal(pred.ravel() < 0.5, te_y)

fpr,tpr,_ = sklearn.metrics.roc_curve(te_y,probs)
roc_auc = sklearn.metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig("ROC.png")

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))



print(te_pairs[0:,0])



# x_eval = []
# y_eval = []

# for target in img_groups:
#     for img_file in img_groups[target]:
#         if int(target) >= 8:
#             img = cv2.imread(os.path.join(image_dir, target, img_file))
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             x_eval.append(img)
#             y_eval.append(int(target))

# x_eval = np.array(x_eval)
# y_eval = np.array(y_eval)
# x_eval = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# x_eval = x_train.astype('float32')
# x_eval /= 255

# digit_indices = [np.where(y_eval == i)[0] for i in range(8,10)]
# eval_pairs, ev_y = create_eval_pairs(x_eval, digit_indices)

# pred = model.predict([eval_pairs[:, 0], eval_pairs[:, 1]])
# eval_acc = compute_accuracy(ev_y, pred)
# print('* Accuracy on evaluation set: %0.2f%%' % (100 * eval_acc))





# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer(name="model_1_1/final/Relu").output)
# intermediate_output = intermediate_layer_model.predict([te_pairs[:, 0], te_pairs[:, 1]])

# np.savetxt("intermediate_output1.csv",vectors[-6000:],delimiter=",")

# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer(name="final/Relu").output)
# intermediate_output = intermediate_layer_model.predict([te_pairs[:, 0], te_pairs[:, 1]])

# np.savetxt("intermediate_output2.csv",vectors[-6000:],delimiter=",")






save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'openWorld.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# sess = K.get_session()

# #[print(x.op.name) for x in model.outputs]

# constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(),['lambda_1/Sqrt'])
# tf.train.write_graph(constant_graph, "", "graph.pb", as_text=False)

# import cv2 as cv

# net = cv.dnn.readNetFromTensorflow('graph.pb')
# print("Success")