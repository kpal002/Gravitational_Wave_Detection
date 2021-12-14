import tensorflow as tf
import numpy as np
import os
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.layers import Input, Add, Dense, Activation, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from matplotlib.pyplot import imshow


def identity_block(X, filter):
    """
    Implementation of the identity block that will be used for the skip connection
    
    Arguments:
    X -- input tensor whose dimension depend on the shape of the image in the previous layer.
    filter -- filter size

    """
    X_skip = X
    # Layer 1
    X = Conv2D(filter, (3,3), padding = 'same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # Layer 2
    X = Conv2D(filter, (3,3), padding = 'same')(X)
    X = BatchNormalization(axis=3)(X)
    # Add Residue
    X = Add()([X, X_skip])     
    X = Activation('relu')(X)
    return X



def convolutional_block(X, filter):
    """
    Implementation of the convolutional block
    
    Arguments:
    X -- input tensor whose dimension depend on the shape of the image in the previous layer.
    
    filter -- filter size
    
    """
    X_skip = X
    # Layer 1
    X = Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    # Layer 2
    X = Conv2D(filter, (3,3), padding = 'same')(X)
    X = BatchNormalization(axis=3)(X)
    # Processing Residue with conv(1,1)
    X_skip = Conv2D(filter, (1,1), strides = (2,2))(X_skip)
    # Add Residue
    X = Add()([X, X_skip])     
    X = Activation('relu')(X)
    return X


def ResNet34(shape):
    # First step is the Input

    X_input = Input(shape)
    X = Conv2D(3, (2, 1), activation='relu' , padding='valid')(X_input)
    X = ZeroPadding2D((3, 3))(X)
    # First Convolutional layer with MaXpooling

    X = Conv2D(64, kernel_size=7, strides=2, padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=3, strides=2, padding='same')(X)
    # sub-blocks and initial filter size


    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For the first sub-block there is no Convolutional block
            for j in range(block_layers[i]):
                X = identity_block(X, filter_size)
        else:
            # One Residual/Convolutional Block followed by [ block_layers - 1] identity blocks
            # The filter size keep increasing by a factor of 2
            filter_size = filter_size*2
            X = convolutional_block(X, filter_size)
            for j in range(block_layers[i] - 1):
                X = identity_block(X, filter_size)
    
    # Step 4 End Dense Network
    X = AveragePooling2D((2,2), padding = 'same')(X)
    X = Flatten()(X)
    X = Dense(256, activation = 'relu')(X) # Not part of the traditional Resnet34 architecture
    X = Dropout(0.2)(X)
    X = Dense(512, activation = 'relu')(X) # Not part of the traditional Resnet34 architecture
    X = Dropout(0.2)(X)
    X = Dense(1, activation = 'sigmoid')(X)
    model = Model(inputs = X_input, outputs = X, name = "ResNet34")
    return model

