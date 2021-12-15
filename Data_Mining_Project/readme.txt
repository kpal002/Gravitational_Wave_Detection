All the neural network models are implemented on tensorflow and keras (2.7.0)

The dataset used for the competition is taken from : https://www.kaggle.com/c/g2net-gravitational-wave-detection/data.
There are 560000 training samples with labels 0 or 1 and 226000 test samples without labels.
Each times series is a numpy array of shape (3,4096) for data from three detectors for 2 sec sampled at 2048 Hz.

The training and testing data is coverted to HD5 format as it leads to efficient storage and loading in Tensorflow. The code for that is available in conversion.py training_labels.csv is required which is attached.

All the normalization, whitening, filtering and transformations acan be found in Preprocessing.py. Inside that, the preprocess() function needs to be modified to use and apply different transformations to time series data. Different transformations will lead to different output shapes based on the parameters used for the project

1. GAF - Gramian angular fields (3 X 64 X 64). 64 X 64 images and 3 for 3 different time series.

2. MTF - Markov TRansition Fields (3 X 64 X 64). 6d X 64 images and 3 for 3 different time series.

3. CQT_transform - (3 X 64 X 65) images

4. VQT transform - (3 X 128 X 129) images (not used)

5. STFT transform - (3 X 257 X 65) images 


Next there are 5 model files

1. CNN.py
2. RESNET34.py
3. RESNET50.py
4. RNN_FCN.py
5. Encoder.py

The squeeze_and_excite() function inside RNN_FCN is not my own code and taken from https://github.com/titu1994/MLSTM-FCN. The rest of the code is written completely by me though I frequently visited https://keras.io/examples/  and https://www.tensorflow.org/tutorials for help. 


finally, there is train+test.py for training and evaluating all the 5 models. All the models are imported there and only needs to be called. The first three models require the input shape of images as input (because of the varying shape of images). For Encoder and RNN_FCN, no input parameters are required.

Once the data is available in HD5 format (either as time series or images), tensorflow datasets will be created and fed to individual models, the model trains and a csv file is generated after evaluating the test data which can be directly upload to kaggle competition website to get the final ROC-AUC scores.

All the csv files used to obtain the AUC scores are included as well as all the models that gave the best score in HD5 format and also all the plots of accuracy, loss and ROC-AUC.

