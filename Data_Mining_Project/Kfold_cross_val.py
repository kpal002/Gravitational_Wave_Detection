import tensorflow as tf
import numpy as np
import h5py
import csv
from os import listdir
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from tensorflow.keras.utils import to_categorical
import keras
import keras_metrics
from tensorflow.keras.optimizers import Adam
from CNN import simple_CNN
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt


# Define the K-fold Cross 
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)


h5f_train = h5py.File('training_cqt.h5','r')
train_data = h5f_train.get('train_data')
train_label = h5f_train.get('train_label')


#optimizers

opt = tf.keras.optimizers.SGD(lr=0.005, decay=1e-6, momentum = 0.9, nesterov=True)
adam_fine = Adam(learning_rate=0.00002, beta_1=0.95, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #smaller than standard
batch_size = 64
no_epochs = 20
for train, test in kfold.split(train_data, train_label):
	n_fold = 1
	model = simple_CNN(input_shape = (64, 65, 3))
	loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
	callbacks = [
        keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_auc"
        ),
        keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001
        ),
	]
	model.compile(optimizer=opt,loss=loss,metrics=[tf.keras.metrics.AUC(),'accuracy'])
	history = model.fit(np.moveaxis(train_data[train],1,3),train_label[train],
            batch_size=batch_size,
            epochs=no_epochs,
            callbacks = callbacks,
            verbose=verbosity)
        
	y_pred = model.evaluate(train_data[test], verbose=1)
	TN, FP, FN, TP = confusion_matrix(train_label[test],y_pred).ravel()
	accuracy = TP+TN/TP+FP+FN+TN
	precision = TP/TP+FP
	recall = TP/TP+FN
	f1_score = 2*(recall * precision) / (recall + precision)

	print('KFold='+str(n_fold))
	print('confusion_matrix : ')
	print(confusion_matrix(train_label[test],y_pred))
	print('Accuracy :' ,accuracy)
	print('precision :' ,precision)
	print('Recall :' ,recall)
	print('F1_score :' ,f1_score)
	print("\n")



	n_fold += 1
