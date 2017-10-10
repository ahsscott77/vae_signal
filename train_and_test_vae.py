#!/usr/bin/python3
#import tensorflow as tf
import numpy as np
from scipy import signal
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import pickle
import vae
import os
import waveform_generator

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
#     nyq =  fs/2.0
#     low = lowcut / nyq
#     high = highcut / nyq
#     #print('lowcut '+repr(lowcut))
#     #print('highcut '+repr(highcut))
#     #print('low ' + repr(low))
#     #print('high ' + repr(high))
#     #y=np.zeros(data.shape)
#     try:
#         b, a = signal.butter(order, [low, high], btype='band')
#         y = signal.lfilter(b, a, data)
#     except ValueError:
#         print('barf')
#     return y
#
#
#
# def spectrogram(x, lowcut, highcut,fs,int_time=512,novrlp=256):
#     y = butter_bandpass_filter(x, lowcut, highcut, fs, order=5)
#     #noise_power = n_p * fs / 2
#     #y += np.random.normal(scale=np.sqrt(noise_power), size=len(y))
#     f, t, Sxx = signal.spectrogram(y, fs=fs, window='hamming', nperseg=int_time, noverlap=novrlp)
#     n = 84
#
#     Sxx_re = misc.imresize(Sxx, (n, n))
#     IMAGE_SIZE=(84,84)
#     Sxx_re=cv2.resize(Sxx_re, IMAGE_SIZE[::-1])
#     cmap = plt.get_cmap('jet')
#     rgba_img = cmap(Sxx_re)
#     rgb_img = np.delete(rgba_img, 3, 2)
#     Sxx_re = cv2.resize(rgb_img, IMAGE_SIZE[::-1])
#     return Sxx_re

network_architecture_2_300_200_100 = \
    dict(n_hidden_recog_1=300, # 1st layer encoder neurons
         n_hidden_recog_2=200, # 2nd layer encoder neurons
         n_hidden_recog_3=100, # 3rd layer encoder neurons
         n_hidden_gener_1=100, # 1st layer decoder neurons
         n_hidden_gener_2=200, # 2nd layer decoder neurons
         n_hidden_gener_3=300,  # 3rd layer decoder neurons
         n_input=1024,
         n_z=2)
#vae_low_noise=vae_3
#vae_3 = train(network_architecture, training_epochs=50)

# network_architecture_mnist = \
#     dict(n_hidden_recog_1=500, # 1st layer encoder neurons
#          n_hidden_recog_2=500, # 2nd layer encoder neurons
#          n_hidden_gener_1=500, # 1st layer decoder neurons
#          n_hidden_gener_2=500, # 2nd layer decoder neurons
#          n_input=28*28, # MNIST data input (img shape: 28*28)
#          n_z=20)  # dimensionality of latent space


#waveform_generator.write_out_data('train')
#waveform_generator.write_out_data('test')
#could make these arguments
train_data_prefix='waveform_data_fs_10000_10000_data_points_nonoverlapping_low_noise_'
test_data_prefix='waveform_data_fs_10000_100_data_points_nonoverlapping_low_noise_channel_'

vae_low_noise_2_300_200_100 = vae.train(network_architecture_2_300_200_100, training_epochs=50,num_layers=3,
                                    file_prefix=train_data_prefix)


test_case=13
plt.figure()
for t in [0,1,2,3]:
    waveform_data = pickle.load(open(test_data_prefix+repr(t)+'.out','rb'))
    max_val_signal = waveform_data[2]
    #read in 100 signal of each type
    batch_xs_test=vae.get_waveform_batch(waveform_data[0],0,100,0.0)
    #get the latent code
    vae_codes = vae_low_noise_2_300_200_100.transform(batch_xs_test)
    #this shows that the latent codes are distinct for the different waveforms
    #even with multipath distortion when the training data didnt habe multipath
    #we could then make a trivial classifier on the codes
    #haven't done the detection part yet

    for i in range(2):
        plt.subplot(4,1,t+1)
        plt.plot(vae_codes[:, i])

        #plt.ylim((-30,30))
plt.show()