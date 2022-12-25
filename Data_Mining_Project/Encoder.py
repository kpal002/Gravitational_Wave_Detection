# Import TensorFlow and other necessary libraries
import tensorflow as tf
import numpy as np
import h5py
import os
import keras
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

# Import the preprocess_input and decode_predictions functions from the ResNet v2 model
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions

# Import layers from TensorFlow
from tensorflow.keras import layers
from tensorflow.python.framework.ops import EagerTensor

# Import various layers and operations from Keras
from keras.layers import Input, Add, Dense, Activation, PReLU, Dropout, ZeroPadding2D, Lambda, Softmax, Flatten, Conv1D, MaxPooling1D, Multiply

# Import the Model and load_model classes from TensorFlow's Keras API
from tensorflow.keras.models import Model, load_model

# Import various initializers from TensorFlow's Keras API
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity

# Import the imshow function from matplotlib
from matplotlib.pyplot import imshow


def Encoder(input_shape= (4096,3)):
    # Define the input layer with the given input shape
    input_layer = Input(input_shape)

    # Add a 1D convolutional layer with 128 filters, a kernel size of 5, and a stride of 1
    # This layer has same padding and is followed by an instance normalization layer and a PReLU activation function
    x1 = Conv1D(128, kernel_size=5, strides=1, padding='same')(input_layer)
    x1 = tfa.layers.InstanceNormalization()(x1)
    x1 = PReLU(shared_axes=[1])(x1)

    # Add a dropout layer with a rate of 0.2, which randomly sets 20% of the input tensor's elements to zero during training to prevent overfitting
    x1 = Dropout(0.2)(x1)

    # Add a max pooling layer with a pool size of 2 that down-samples the output by taking the maximum value of every two consecutive elements
    x1 = MaxPooling1D(pool_size=2)(x1)

    # Add another 1D convolutional layer with 256 filters, a kernel size of 11, and a stride of 1
    # This layer has same padding and is followed by an instance normalization layer and a PReLU activation function
    x2 = Conv1D(256, kernel_size=11, strides=1, padding='same')(x1)
    x2 = tfa.layers.InstanceNormalization()(x2)
    x2 = PReLU(shared_axes=[1])(x2)

    # Add another dropout layer with a rate of 0.2, which randomly sets 20% of the input tensor's elements to zero during training to prevent overfitting
    x2 = Dropout(rate=0.2)(x2)

    # Add another max pooling layer with a pool size of 2 that down-samples the output by taking the maximum value of every two consecutive elements
    x2 = MaxPooling1D(pool_size=2)(x2)

    
    # Add another 1D convolutional layer with 512 filters, a kernel size of 21, and a stride of 1
    # This layer has same padding and is followed by an instance normalization layer and a PReLU activation function
    # Notice that there is no max pooling layer after this convolutional block
    conv3 = Conv1D(512, kernel_size=21, strides=1, padding='same')(x2)
    conv3 = tfa.layers.InstanceNormalization()(conv3)
    conv3 = PReLU(shared_axes=[1])(conv3)

    # Add another dropout layer with a rate of 0.2, which randomly sets 20% of the input tensor's elements to zero during training to prevent overfitting
    conv3 = Dropout(rate=0.2)(conv3)

    # Split the output tensor into two parts along the channel axis (axis=2) using the Lambda layer
    # The first part is used as attention data and the second part is passed through a softmax function to obtain attention weights
    attention_data = Lambda(lambda x: x[:,:,:256])(conv3)
    attention_softmax = Lambda(lambda x: x[:,:,256:])(conv3)

    # Apply the softmax function to the attention weights to normalize them
    attention_softmax = Softmax()(attention_softmax)

    # Multiply the attention data and the attention weights element-wise using the Multiply layer
    multiply_layer = Multiply()([attention_softmax, attention_data])

    # Add a dense layer with 512 units and a sigmoid activation function
    # This layer is followed by an instance normalization layer
	
    # Add a dense layer with 512 units and a sigmoid activation function
    # This layer is followed by an instance normalization layer
    dense_layer = Dense(units=512, activation='sigmoid')(multiply_layer)
    dense_layer = tfa.layers.InstanceNormalization()(dense_layer)

    # Flatten the output of the dense layer to a 1D tensor
    flatten_layer = Flatten()(dense_layer)

    # Add a final dense layer with 1 unit and a sigmoid activation function to obtain the output
    output_layer = Dense(1, activation='sigmoid')(flatten_layer)

    # Create a model using the input layer and output layer
    model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='Encoder')

    # Return the model
    return model
