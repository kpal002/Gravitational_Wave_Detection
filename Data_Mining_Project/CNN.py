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
import frankwolfe.tensorflow as fw
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
#from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import SGD, RMSprop
import matplotlib.pyplot as plt


# Convolutional layer and maxpool layer 1


def simple_CNN(input_shape = (65, 64, 3)):
    X_input = Input(input_shape)
    X = Conv2D(3, (2, 1), activation='relu' , padding='valid', input_shape=(65, 64,3))(X_input) # Change the iput dimension to 64 x 64.


    X = Conv2D(64, (3, 3), activation='relu' , padding='same')(X)
    X = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D((2, 2))(X)



    X = Conv2D(128, (3, 3), activation='relu',  padding='same')(X)
    X = Conv2D(128, (3, 3), activation='relu', padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = MaxPooling2D((2, 2))(X)
    
    X = Dropout(0.2)(X)
    X = Flatten()(X)
    X = Dense(512, activation='relu')(X)
    X = Dropout(0.3)(X)
    X_output = Dense(1, activation='sigmoid')(X)


    # Create model
    model = Model(inputs = X_input, outputs = X_output, name = 'CNN')

    return model


