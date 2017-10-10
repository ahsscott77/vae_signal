import numpy as np
from scipy import signal

from scipy import misc
import cv2
import matplotlib.pyplot as plt
import pickle

def butter_bandpass_filter(data, lowcut, highcut, fs, order=9):
    nyq =  fs/2.0
    low = lowcut / nyq
    high = highcut / nyq
    #print('lowcut '+repr(lowcut))
    #print('highcut '+repr(highcut))
    #print('low ' + repr(low))
    #print('high ' + repr(high))
    #y=np.zeros(data.shape)
    try:
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.lfilter(b, a, data)
    except ValueError:
        print('barf')
    return y



def spectrogram(x, lowcut, highcut,fs,int_time=512,novrlp=256):
    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=5)
    #noise_power = n_p * fs / 2
    #y += np.random.normal(scale=np.sqrt(noise_power), size=len(y))
    f, t, Sxx = signal.spectrogram(y, fs=fs, window='hamming', nperseg=int_time, noverlap=novrlp)
    n = 84

    Sxx_re = misc.imresize(Sxx, (n, n))
    IMAGE_SIZE=(84,84)
    Sxx_re=cv2.resize(Sxx_re, IMAGE_SIZE[::-1])
    cmap = plt.get_cmap('jet')
    rgba_img = cmap(Sxx_re)
    rgb_img = np.delete(rgba_img, 3, 2)
    Sxx_re = cv2.resize(rgb_img, IMAGE_SIZE[::-1])
    return Sxx_re




def generate_data(type,fc,bw,N,n_p,n_mean,fs):
    #create a signal by adding a waveform to noise and later interference
    amp = 2 * np.sqrt(2)
    noise_power = n_p * fs / 2
    #noise_power = 0.01 * fs / 2
    #print(type)
    if type==0:
        # upsweep
        # time = np.arange(N) / fs
        # freq = np.linspace(1e3, 2e3, N)
        time = np.arange(N) / 1.0 / fs
        #freq = np.linspace(1e3, 1.5e3, N)
        fstart=1e3
        fstop=1.5e3
        k=(fstop-fstart)/(np.round(N * 0.8)/ fs)
        x = amp * np.cos(2 * np.pi * (k/2.0*time*time+fstart*time))
        x[0:np.round(N * 0.1).astype('int32')] = 0
        x[np.round(N * 0.9).astype('int32'):N] = 0
        # add noise
        x += n_mean+np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

        # lowcut = 1e3
        # highcut = 2e3
        #
        # template = spectrogram(x, lowcut, highcut,fs,256,128)


    elif type==1:

        # #downsweep
        time = np.arange(N) / 1.0 / fs
        fstart = 4e3
        fstop = 2.5e3
        k = (fstop - fstart) / (np.round(N * 0.8) / fs)
        x = amp * np.cos(2 * np.pi * (k / 2.0 * time * time + fstart * time))
        x[0:np.round(N * 0.1).astype('int32')] = 0
        x[np.round(N * 0.9).astype('int32'):N] = 0
        # add noise
        x += n_mean+np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

        # lowcut = 2e3
        # highcut = 4e3
        #
        # template = spectrogram(x, lowcut, highcut,fs,128,64)

    elif type==2:

        # tone
        time = np.arange(N) / 1.0 / fs
        freq = np.ones(time.shape) * 500
        x = amp * np.sin(2 * np.pi * freq * time)
        x[0:np.round(N * 0.1).astype('int32')] = 0
        x[np.round(N * 0.9).astype('int32'):N] = 0
        # add noise
        x += n_mean+np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

        # lowcut = 2900
        # highcut = 3100

        #template = spectrogram(x, lowcut, highcut,fs,512,256)
    elif type==3:
        time = np.arange(N) / 1.0 / fs
        x = n_mean + np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    return x


def generate_multipath_channel(num_paths,ratio_direct_to_others,N):
    #doing real amplitudes, no phase change
    #set the direct path to one and use the ratio to set the other
    tap1=1
    time_delays=np.sort(np.round(np.random.uniform(0.01,.25,num_paths)*N))
    amps = np.exp(-1 * time_delays / N)
    #putting more of the mass on positive, negative is 180 degree phase shift
    random_pos_neg=np.sign(np.random.uniform(-1,2,num_paths))
    amps=amps*random_pos_neg
    #/(sum(amps)*ratio_direct_to_others)
    channel=np.zeros(N,)
    channel[0]=tap1
    for p in range(num_paths):
        channel[int(time_delays[p])]=amps[p]
    channel=channel/np.linalg.norm(channel)
    denom=np.hstack((1,np.zeros(N-1,)))
    return channel,denom

N=1024
fs=10000
for t in [0,1,2,3]:
    signal_dict = dict()
    fft_dict = dict()
    #template_dict = dict()
    #spectro_dict = dict()
    max_val_signal = 0
    max_val_fft = 0
    n_p_vec = np.random.uniform(1e-3,1e-2,100)
    for m in [0]:#-1,-0.5,0.0,0.5,1.0
        print(m)
        for i in range(100):
            bw = np.round(np.random.randint(200,1000))
            fc = np.round(np.random.randint(1100,2900))
            int_time = 2**np.random.randint(6,10)
            ovrlp = int_time/2
            lowcut = fc - bw / 2.0

            #n_p=n_p_vec[i]
            highcut = fc + bw / 2.0
            # if t==3:
            #     n_p=np.random.uniform()/100.0
            # else:
            n_p=1e-5
            num_paths = int(np.round(np.random.uniform(0.5,5.49)))
            power_ratio = np.random.uniform(1.5,3.5)
            x=generate_data(t,fc,bw,N,n_p,m,fs)

            channel1, denom1 = generate_multipath_channel(num_paths, power_ratio, N)
            x=signal.filtfilt(channel1,denom1,x,padlen=256)
            # plt.figure()
            # plt.imshow(template)
            # plt.show()
            #
            # plt.figure()
            # plt.plot(x)
            # plt.show()

            #spectro=spectrogram(x, lowcut, highcut, fs, int_time=int_time, novrlp=ovrlp)
            max_x=np.amax(np.abs(x))
            if max_x>max_val_signal:
                max_val_signal=max_x

            signal_dict[(m,i)]=x

            Y = np.fft.fft(x)
            max_Y = np.max((np.amax(np.real(Y)),np.amax(np.imag(Y))))
            if max_Y > max_val_fft:
                max_val_fft = max_Y

            fft_dict[(m, i)] = Y
            #template_dict[(m,i)]=template
            #spectro_dict[(t,m,i)]=(fc,bw,spectro)
    pickle.dump((signal_dict,fft_dict,max_val_signal,max_val_fft),open('waveform_data_fs_10000_100_data_points_nonoverlapping_low_noise_channel_'+repr(t)+'.out','wb'))

