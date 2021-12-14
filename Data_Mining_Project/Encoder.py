import tensorflow as tf
import numpy as np
import h5py
import os
import keras
import tensorflow_io as tfio
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.python.framework.ops import EagerTensor
from keras.layers import Input, Add, Dense, Activation, PReLU, Dropout, ZeroPadding2D, Lambda, Softmax, Flatten, Conv1D, MaxPooling1D, Multiply
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
from matplotlib.pyplot import imshow


def Encoder(input_shape= (4096,3)):
	input_layer = Input(input_shape)

	# conv block 1
	x1 = Conv1D(128,kernel_size=5,strides=1,padding='same')(input_layer)
	x1 = tfa.layers.InstanceNormalization()(x1)
	x1 = PReLU(shared_axes=[1])(x1)
	x1 = Dropout(0.2)(x1)
	x1 = MaxPooling1D(pool_size=2)(x1)
 	
 	# conv block 2
	x2 = Conv1D(256,kernel_size=11,strides=1,padding='same')(x1)
	x2 = tfa.layers.InstanceNormalization()(x2)
	x2 = PReLU(shared_axes=[1])(x2)
	x2 = Dropout(rate=0.2)(x2)
	x2 = MaxPooling1D(pool_size=2)(x2)
    
    # conv block 3 (Motice No MaxPooling)
	conv3 = Conv1D(512,kernel_size=21,strides=1,padding='same')(x2)
	conv3 = tfa.layers.InstanceNormalization()(conv3)
	conv3 = PReLU(shared_axes=[1])(conv3)
	conv3 = Dropout(rate=0.2)(conv3)
    
    # split for attention
	attention_data = Lambda(lambda x: x[:,:,:256])(conv3)
	attention_softmax = Lambda(lambda x: x[:,:,256:])(conv3)
    
    # attention mechanism
	attention_softmax = Softmax()(attention_softmax)
	multiply_layer = Multiply()([attention_softmax,attention_data])
	
	# last layer
	dense_layer = Dense(units=512,activation='sigmoid')(multiply_layer)
	dense_layer = tfa.layers.InstanceNormalization()(dense_layer)
	
	# output layer
	flatten_layer = Flatten()(dense_layer)
	output_layer = Dense(1,activation='sigmoid')(flatten_layer)

	model = keras.models.Model(inputs=input_layer, outputs=output_layer, name='Encoder')

	return model