import tensorflow as tf
import numpy as np
import h5py
import csv
from os import listdir
from tensorflow import keras
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import keras
import keras_metrics
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt


batch_size = 64
img_width, img_height, img_num_channels = 64, 65, 3
no_epochs = 50
validation_split = 0.1
verbosity = 1


h5f_train = h5py.File('training_cqt.h5','r')
train_data = h5f_train['train_data'][:]
train_label = h5f_train['train_label'][:]
val_data = h5f_train['val_data'][:]
val_label = h5f_train['val_label'][:]
h5f_train.close()

print(train_data.shape)
m = train_data.shape[0]
m_val = val_data.shape[0]

#train_data = train_data.reshape((m, img_width, img_height, img_num_channels))
#val_data  = val_data.reshape((m_val, img_width, img_height, img_num_channels))

# Convolutional layer and maxpool layer 1


def simple_CNN(input_shape = (64, 65, 3)):
    X_input = Input(input_shape)
    X = Conv2D(3, (1, 2), activation='relu' , padding='valid',input_shape=(64, 65,3))(X_input)
    X = Conv2D(32, (3, 3), activation='relu' , padding='same')(X)
    X = Conv2D(32, (3, 3), activation='relu', padding='same')(X)
    X = MaxPooling2D((2, 2))(X)


    X = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
    X = Conv2D(64, (3, 3), activation='relu',  padding='same')(X)
    X = MaxPooling2D((2, 2))(X)


    X = Conv2D(128, (3, 3), activation='relu',  padding='same')(X)
    X = Conv2D(128, (3, 3), activation='relu', padding='same')(X)
    X = MaxPooling2D((2, 2))(X)
    
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.2)(X)
    X_output = Dense(1, activation='sigmoid')(X)


    # Create model
    model = Model(inputs = X_input, outputs = X_output)

    return model


batch_size = 64

model.summary()
#opt = tf.compat.v1.train.AdamOptimizer(
#    learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-07, use_locking=False,
#    name='Adam'
#)
opt = tf.keras.optimizers.SGD(lr=0.005, decay=1e-6, momentum = 0.9, nesterov=True)
adam_fine = Adam(lr=0.00002, beta_1=0.95, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #20x smaller than standard
#loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
#model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])

model.compile(optimizer=adam_fine,loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(),'accuracy'])
history = model.fit(np.moveaxis(train_data,1,3),train_label,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_data=(val_data,val_label))

plt.figure(figsize=(10,10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy.pdf')

plt.figure(figsize=(10,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss.pdf')
