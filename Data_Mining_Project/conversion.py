import h5py # Store data files in .h5 format
import numpy as np
import pandas as pd
from glob import glob # Read data files stored in pathnames following a pattern.
import matplotlib.pyplot as plt


train_data_list , train_label_list = [],[]
test_data_list = []

# Read training label from the CSV file

df_train_label = pd.read_csv('training_labels.csv')



# read datfiles stored in folders based on the properties of BH signal
paths_files = glob("test/*/*/*/*")
paths_files.sort()

print(paths_files[:10]) # To check everything is ok

train_data = np.load(paths_files[0])


for i in range(560000):
	train_data_list.extend([np.load(paths_files[i]).T])
	train_label_list.append(df_train_label['target'][i])


train_data = np.array(train_data_list)
train_label = np.array(train_label_list)
train_label.reshape((560000,1))
del train_data_list
del train_label_list

# Check for the correct shapes
print(train_data.shape)
print(train_label.shape)

# Store as HD5 file
h5f1 = h5py.File('training_trial.h5', 'w')
h5f1.create_dataset('train_data', data=train_data)
h5f1.create_dataset('train_label', data=train_label)
h5f1.close()