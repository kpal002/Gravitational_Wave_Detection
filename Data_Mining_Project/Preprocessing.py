import h5py
import time
import torch
from torch.fft import fft, rfft, ifft
import librosa
import numpy as np
from multiprocessing import Pool
from scipy import signal
from numba import jit, prange
import matplotlib.pyplot as plt
from nnAudio.Spectrogram import CQT1992v2, STFT
from numba.experimental import jitclass
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField, RecurrencePlot



def normalize(time_series):
        
    # Scaling the data in the [-1,1] range so as to transform to polar co-ordinates
    # for Gramian angular fields.

    min_stamp = np.amin(time_series)
    max_stamp = np.amax(time_series)
    time_series = (2*time_series - max_stamp - min_stamp)/(max_stamp - min_stamp)

    # Checking for any floating point accuracy.
    time_series = np.where(time_series >= 1., 1., time_series)
    time_series = np.where(time_series <= -1., -1., time_series)

    return time_series


def whiten(x):
    hann = torch.hann_window(len(x), periodic=True, dtype=float)
    spec = fft(torch.from_numpy(x).float()* hann)
    mag = torch.sqrt(torch.real(spec*torch.conj(spec))) 

    return np.array(torch.real(ifft(spec/mag)).numpy() * np.sqrt(len(x)/2))
    

# Apply a butterworth bandpass filter of order 8 and within the range [20,500]. 
# The sampling rate for the data is 2048. the Nyquist frequency is 1024. 

def apply_bandpass(x, lf=20, hf=500, order=8, sr=2048):
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))

    x *= signal.tukey(4096, 0.2)
    x = signal.sosfiltfilt(sos, x) / normalization
    return x

# CQT transform 
def CQT_transform(x):
    return np.abs(librosa.cqt(x, sr=2048, fmin = 20, hop_length=64, bins_per_octave=24, n_bins=64, tuning=0.0, filter_scale=1.0, norm=1, sparsity=0.01, 
        window='hann', scale=True, pad_mode='reflect'))



def VQT_transform(x):

    return np.abs(librosa.vqt(x, sr=2048, fmin = 20, hop_length=32, n_bins=128, bins_per_octave=32))

def STFT_transform(x):
    return np.abs(librosa.stft(x, n_fft=512, hop_length=64, win_length=512, window='hann', center=True, dtype=None, pad_mode='reflect'))
#)


def cwt_morlet(x):
    return np.abs(signal.cwt(x, signal.morlet, np.arange(1, 10)))






# Gramian Angular Field using both summation and difference method
def GAF(x,img_size,methods):
    gaf = GramianAngularField(image_size = img_size, method = methods)
    image = gaf.fit_transform(x.reshape(1,-1))
        
    return image.T
         
# Markov Field Transition
def MTF(x,img_size):
    mtf = MarkovTransitionField(image_size = img_size)
        
    return (mtf.fit_transform(x.reshape(1,-1))).T


def preprocess(x):
    image = np.zeros((3,513,257))
    for i in range(3):
        x1 = normalize(x.T[i])
        #x1 = whiten(x1)
        x1 = apply_bandpass(x1)
        x1 = STFT_transform(x1)
        image[i] = x1
        
    return image

# Reading from training HD5 file
h5_train = h5py.File('training.h5','r')
train_data = h5_train['train_data'][:1]


train = preprocess(train_data[0])
train_label = h5_train['train_label'][:1]
train_images = []


if __name__ == '__main__':
    start = time.time()
    pool = Pool(2)
    train_images = pool.map(preprocess,train_data)

    
    #x_final = np.concatenate((np.expand_dims(x1, axis=2),np.expand_dims(x2, axis=2), np.expand_dims(x3, axis=2)), axis=2)
    
    print("Finished in: {:.2f}s".format(time.time()-start)) 
    train_images = np.array(train_images)
    print(train_images.shape)

    # Writing to a new HD5 file
    cqt_data = h5py.File('training_cqt.h5', 'w')
    cqt_data.create_dataset('train_data', data=train_images[:504000])
    cqt_data.create_dataset('train_label', data=train_label[:504000])
    cqt_data.create_dataset('val_data', data=train_images[504000:560000])
    cqt_data.create_dataset('val_label', data=train_label[504000:560000])
    



