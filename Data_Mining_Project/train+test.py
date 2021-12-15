import tensorflow as tf
import numpy as np
import h5py
import csv
import pandas as pd
import tensorflow_io as tfio
from tensorflow import keras

# Import models from other files
from CNN import simple_CNN
from RESNET34 import ResNet34
from RESNET50 import ResNet50
from RNN_FCN import RNN_FCN
import matplotlib.pyplot as plt
from SAM_optimizer import SAMModel
from Encoder import Encoder


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.python.framework.ops import EagerTensor
from tensorflow.keras.layers import Input, Add, Dense, Activation, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity


# For GPU distributed training
strategy = tf.distribute.MirroredStrategy()
devices = tf.config.experimental.list_physical_devices('GPU')
epochs = 7
batch_size = 64 * strategy.num_replicas_in_sync


# Reading the dataset from HD5 files in batches and separate into train, validation and test set. Define the location of the data here.
x_val = tfio.IODataset.from_hdf5('path to file location', dataset='/val_data')
y_val = tfio.IODataset.from_hdf5('path to file location', dataset='/val_label')

x_train = tfio.IODataset.from_hdf5('path to file location', dataset='/train_data')
y_train = tfio.IODataset.from_hdf5('path to file location', dataset='/train_label')
x_test = tfio.IODataset.from_hdf5('path to file location', dataset='/test_data')

train = tf.data.Dataset.zip((x_train,y_train)).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
val = tf.data.Dataset.zip((x_val,y_val)).batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
test = tf.data.Dataset.zip(x_test).batch(batch_size, drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)




with strategy.scope():
    # model needs to be defined here
    model = Encoder()


    # Implemented callbacks to save the best model based on 'val_auc' and reduce the learning rate in case
    # 'val_loss' does not improve
    callbacks = [
        keras.callbacks.ModelCheckpoint(
        "best_model_"+model._name+".h5", save_best_only=True, monitor="val_auc", mode = 'max', verbose = 1,
        ),
        keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
        ),
    ]

    # Fine tuned Adam optimizer
    adam_fine = Adam(learning_rate=0.00005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(
        optimizer=adam_fine,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(),'accuracy'],
    )

history = model.fit(
    train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks = callbacks,
    validation_data=val,
    verbose=1,
)

model.save('final_model_'+model._name+'.h5')
# Testing part
best_model = keras.models.load_model("best_model_"+model._name+".h5")
# Predict from the model
yhat = best_model.predict(test)
yhat = list(yhat.flatten())



fields = ['id', 'target']

df = pd.read_csv("sample_submission.csv")

id = list(df['id'])
target = yhat
rows_csv = []

for i in range(226000):
    rows_csv.append([str(id[i]),target[i]])
    
filename = "classification_"+model._name+"_final.csv"

with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
        
    # writing the fields
    csvwriter.writerow(fields)
        
    # writing the data rows
    csvwriter.writerows(rows_csv)


print(history)

plt.figure(figsize=(10,10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('CNN accuracy')
plt.ylabel('accuracy',fontsize=12)
plt.xlabel('epoch',fontsize=12)
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('accuracy_'+model._name+'.pdf')


plt.figure(figsize=(10,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('CNN loss')
plt.ylabel('loss',fontsize=12)
plt.xlabel('epoch',fontsize=12)
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss_'+model._name+'.pdf')

plt.figure(figsize=(10,10))
plt.plot(history.history['auc'])
plt.plot(history.history['val_auc'])
plt.xlabel('epoch',fontsize=12)
plt.title('CNN AUC')
plt.ylabel('ROC_AUC',fontsize=12)
plt.legend(['train', 'validation'], loc='lower right', fontsize ='large')
plt.savefig('ROC_'+model._name+'.pdf')
