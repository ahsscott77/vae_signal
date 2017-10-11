from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import numpy as np
import matplotlib.pyplot as plt
import pickle
import vae

train_data_prefix='waveform_data_fs_10000_10000_data_points_overlapping_low_noise_'
#test_data_prefix='waveform_data_fs_10000_100_data_points_overlapping_low_noise_channel2_'
test_data_prefix='waveform_data_fs_10000_100_data_points_overlapping_mid_noise_'

network_architecture_10_300_200_100 = \
    dict(n_hidden_recog_1=300, # 1st layer encoder neurons
         n_hidden_recog_2=200, # 2nd layer encoder neurons
         n_hidden_recog_3=100, # 3rd layer encoder neurons
         n_hidden_gener_1=100, # 1st layer decoder neurons
         n_hidden_gener_2=200, # 2nd layer decoder neurons
         n_hidden_gener_3=300,  # 3rd layer decoder neurons
         n_input=1024,
         n_z=10)

vae_low_noise_10_300_200_100_scale = vae.train(network_architecture_10_300_200_100, training_epochs=50,num_layers=3,
                                   file_prefix=train_data_prefix)
vae_tmp=vae_low_noise_10_300_200_100_scale

#load in the vae once you figure out how to save them
#train
vae_codes_train=np.zeros([40000,10])
labels_train=np.hstack((0*np.ones(10000,),1*np.ones(10000,),2*np.ones(10000,),3*np.ones(10000,)))

vae_codes_test=np.zeros([400,10])
labels_test=np.hstack((0*np.ones(100,),1*np.ones(100,),2*np.ones(100,),3*np.ones(100,)))
for t in [0,1,2,3]:
    waveform_data = pickle.load(open(train_data_prefix+repr(t)+'.out','rb'))
    #read in 10000 signal of each type
    #scaling each sig
    batch_xs_train=vae.get_waveform_batch(waveform_data[0],0,10000,0.0)
    #get the latent code
    vae_codes_train[t*10000:(t+1)*10000,:] = vae_tmp.transform(batch_xs_train)

    waveform_data = pickle.load(open(test_data_prefix + repr(t) + '.out', 'rb'))
    # read in 100 signal of each type
    # scaling each sig
    batch_xs_test = vae.get_waveform_batch(waveform_data[0], 0, 100, 0.0)
    vae_codes_test[t*100:(t+1)*100,:] = vae_tmp.transform(batch_xs_test)

clf_max_each = RandomForestClassifier(random_state=1,n_estimators=101,n_jobs=-1)#max_depth=20,
clf_max_each.fit(vae_codes_train,labels_train)

type_predict=clf_max_each.predict(vae_codes_test)
type_proba=clf_max_each.predict_proba(vae_codes_test)
confusion_matrix(labels_test,type_predict)

plt.figure()
plt.plot(vae_codes_test)

plt.figure()
plt.plot(vae_codes_train)

plt.figure()
plt.plot((batch_xs_train[:,100]+ max_val_signal) / (max_val_signal * 2))



