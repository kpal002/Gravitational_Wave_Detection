# Import the necessary libraries
import tensorflow as tf
import numpy as np
import h5py  # library for reading and writing data in HDF5 format
import csv  # library for reading and writing CSV files
from os import listdir  # library for interacting with the file system
from tensorflow import keras  # TensorFlow's high-level API for building and training models
from sklearn.model_selection import KFold  # library for cross-validation
from tensorflow.keras.utils import to_categorical  # utility for one-hot encoding of labels
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # utility for image preprocessing
from tensorflow.keras.preprocessing import image  # utility for loading and processing images
import keras  # Keras is a high-level API for building and training deep learning models
import keras_metrics  # library for metrics in Keras
import frankwolfe.tensorflow as fw  # library for the Frank-Wolfe optimization algorithm
from tensorflow.keras.optimizers import Adam  # Adam optimization algorithm
from keras.models import Sequential,Input,Model  # models and layers from Keras
from keras.layers import Dense, Dropout, Flatten  # layers for building neural networks
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization  # layers for building CNNs
#from keras.layers.normalization import BatchNormalization  # batch normalization layer (duplicate import)
from keras.layers.advanced_activations import LeakyReLU  # leaky ReLU activation function
from keras.models import Sequential  # Sequential model (duplicate import)



def simple_CNN(input_shape = (65, 64, 3)):
    # Define the input layer with the given input shape
    X_input = Input(input_shape)

    # Add a 2D convolutional layer with 3 filters of size (2, 1) and a ReLU activation function
    # This layer has valid padding and its input shape is (65, 64, 3)
    X = Conv2D(3, (2, 1), activation='relu', padding='valid', input_shape=(65, 64, 3))(X_input)

    # Add two 2D convolutional layers with 64 filters of size (3, 3) and a ReLU activation function
    # Both layers have same padding
    X = Conv2D(64, (3, 3), activation='relu', padding='same')(X)
    X = Conv2D(64, (3, 3), activation='relu', padding='same')(X)

    # Add a batch normalization layer that normalizes the output tensor along the channel axis (axis=3)
    X = BatchNormalization(axis=3)(X)

    # Add a max pooling layer with a pool size of (2, 2) that down-samples the output by taking the maximum value of each patch of size (2, 2)
    X = MaxPooling2D((2, 2))(X)

    # Add two more 2D convolutional layers with 128 filters of size (3, 3) and a ReLU activation function
    # Both layers have same padding
    X = Conv2D(128, (3, 3), activation='relu', padding='same')(X)
    X = Conv2D(128, (3, 3), activation='relu', padding='same')(X)

    # Add another batch normalization layer that normalizes the output tensor along the channel axis (axis=3)
    X = BatchNormalization(axis=3)(X)

    # Add another max pooling layer with a pool size of (2, 2) that down-samples the output by taking the maximum value of each patch of size (2, 2)
    X = MaxPooling2D((2, 2))(X)

    # Add a dropout layer with a rate of 0.2, which randomly sets 20% of the input tensor's elements to zero during training to prevent overfitting
    X = Dropout(0.2)(X)

    # Add a flatten layer that flattens the output tensor into a single-dimensional tensor
    X = Flatten()(X)

    # Add a dense layer with 512 units and a ReLU activation function
    X = Dense(512, activation='relu')(X)

    # Add another dropout layer with a rate of 0.3, which randomly sets 30% of the input tensor's elements to zero during training to prevent overfitting
    X = Dropout(0.3)(X)

    # Add an output layer with a single unit and a sigmoid activation function
    # This output layer represents the predicted probability of the input belonging to a certain class (in this case, a binary classification problem)
    X_output = Dense(1, activation='sigmoid')(X)

    # Create the model with the input layer and the output layer
    model = Model(inputs = X_input, outputs = X_output, name = 'CNN')

    return model


