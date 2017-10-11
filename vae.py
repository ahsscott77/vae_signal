
import tensorflow as tf
import numpy as np
# from scipy import signal
# from scipy import misc
# import cv2
# import matplotlib.pyplot as plt
import pickle



np.random.seed(0)
tf.set_random_seed(0)




# waveform_data1=pickle.load(open('/home/new_sjohnston/waveform_data_fs_10000_10000_data_points_1.out','rb'))
# max_val_signal1=waveform_data1[3]
# max_val_fft1=waveform_data1[4]

n_samples = 10000

def get_waveform_batch(wave_dict,sample,batch_size,m):
    batch=np.zeros([batch_size,1024])
    for i in range(batch_size):
        sig = wave_dict[m, i + sample]
        max_val = np.max(np.abs(sig))
        batch[i, :] = (sig + max_val) / (2 * max_val)
    return batch

def get_waveform_fft_batch(wave_dict,sample,batch_size,m):
    batch=np.zeros([batch_size,2048])
    for i in range(batch_size):
        fft_data=wave_dict[m,i+sample]
        real_fft_data = np.real(fft_data)
        imag_fft_data = np.imag(fft_data) #[range(int(len(fft_data) / 2))]
        stack_data=np.hstack((real_fft_data,imag_fft_data))
        batch[i,:]=stack_data
    return batch


def xavier_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    # https://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class VariationalAutoencoder(object):
    """ Variation Autoencoder (VAE) with an sklearn-like interface implemented using TensorFlow.

    This implementation uses probabilistic encoders and decoders using Gaussian
    distributions and  realized by multi-layer perceptrons. The VAE can be learned
    end-to-end.

    See "Auto-Encoding Variational Bayes" by Kingma and Welling for more details.
    """

    def __init__(self, network_architecture, transfer_fct=tf.nn.softplus,
                 learning_rate=0.001, batch_size=100,num_layers=2):
        self.network_architecture = network_architecture
        self.transfer_fct = transfer_fct
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, network_architecture["n_input"]])

        # Create autoencoder network
        if num_layers==2:
            self._create_network()
        elif num_layers==3:
            self._create_network_3()
        else:
            print('barf, only 2 or 3 layers allowed')

        # Define loss function based variational upper-bound and
        # corresponding optimizer
        self._create_loss_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def _create_network(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def _create_network_3(self):
        # Initialize autoencode network weights and biases
        network_weights = self._initialize_weights_3(**self.network_architecture)

        # Use recognition network to determine mean and
        # (log) variance of Gaussian distribution in latent
        # space
        self.z_mean, self.z_log_sigma_sq = \
            self._recognition_network_3(network_weights["weights_recog"],
                                      network_weights["biases_recog"])

        # Draw one sample z from Gaussian distribution
        n_z = self.network_architecture["n_z"]
        eps = tf.random_normal((self.batch_size, n_z), 0, 1,
                               dtype=tf.float32)
        # z = mu + sigma*epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))

        # Use generator to determine mean of
        # Bernoulli distribution of reconstructed input
        self.x_reconstr_mean = \
            self._generator_network_3(network_weights["weights_gener"],
                                    network_weights["biases_gener"])

    def _initialize_weights(self, n_hidden_recog_1, n_hidden_recog_2,
                            n_hidden_gener_1, n_hidden_gener_2,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_2, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_2, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_2, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_2, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _initialize_weights_3(self, n_hidden_recog_1, n_hidden_recog_2, n_hidden_recog_3,
                            n_hidden_gener_1, n_hidden_gener_2, n_hidden_gener_3,
                            n_input, n_z):
        all_weights = dict()
        all_weights['weights_recog'] = {
            'h1': tf.Variable(xavier_init(n_input, n_hidden_recog_1)),
            'h2': tf.Variable(xavier_init(n_hidden_recog_1, n_hidden_recog_2)),
            'h3': tf.Variable(xavier_init(n_hidden_recog_2, n_hidden_recog_3)),
            'out_mean': tf.Variable(xavier_init(n_hidden_recog_3, n_z)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_recog_3, n_z))}
        all_weights['biases_recog'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_recog_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_recog_2], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_recog_3], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_z], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_z], dtype=tf.float32))}
        all_weights['weights_gener'] = {
            'h1': tf.Variable(xavier_init(n_z, n_hidden_gener_1)),
            'h2': tf.Variable(xavier_init(n_hidden_gener_1, n_hidden_gener_2)),
            'h3': tf.Variable(xavier_init(n_hidden_gener_2, n_hidden_gener_3)),
            'out_mean': tf.Variable(xavier_init(n_hidden_gener_3, n_input)),
            'out_log_sigma': tf.Variable(xavier_init(n_hidden_gener_3, n_input))}
        all_weights['biases_gener'] = {
            'b1': tf.Variable(tf.zeros([n_hidden_gener_1], dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden_gener_2], dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden_gener_3], dtype=tf.float32)),
            'out_mean': tf.Variable(tf.zeros([n_input], dtype=tf.float32)),
            'out_log_sigma': tf.Variable(tf.zeros([n_input], dtype=tf.float32))}
        return all_weights

    def _recognition_network(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        z_mean = tf.add(tf.matmul(layer_2, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_2, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _recognition_network_3(self, weights, biases):
        # Generate probabilistic encoder (recognition network), which
        # maps inputs onto a normal distribution in latent space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.x, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2, weights['h3']),
                                           biases['b3']))
        z_mean = tf.add(tf.matmul(layer_3, weights['out_mean']),
                        biases['out_mean'])
        z_log_sigma_sq = \
            tf.add(tf.matmul(layer_3, weights['out_log_sigma']),
                   biases['out_log_sigma'])
        return (z_mean, z_log_sigma_sq)

    def _generator_network_3(self, weights, biases):
        # Generate probabilistic decoder (decoder network), which
        # maps points in latent space onto a Bernoulli distribution in data space.
        # The transformation is parametrized and can be learned.
        layer_1 = self.transfer_fct(tf.add(tf.matmul(self.z, weights['h1']),
                                           biases['b1']))
        layer_2 = self.transfer_fct(tf.add(tf.matmul(layer_1, weights['h2']),
                                           biases['b2']))
        layer_3 = self.transfer_fct(tf.add(tf.matmul(layer_2, weights['h3']),
                                           biases['b3']))
        x_reconstr_mean = \
            tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['out_mean']),
                                 biases['out_mean']))
        return x_reconstr_mean

    def _create_loss_optimizer(self):
        # The loss is composed of two terms:
        # 1.) The reconstruction loss (the negative log probability
        #     of the input under the reconstructed Bernoulli distribution
        #     induced by the decoder in the data space).
        #     This can be interpreted as the number of "nats" required
        #     for reconstructing the input when the activation in latent
        #     is given.
        # Adding 1e-10 to avoid evaluation of log(0.0)
        reconstr_loss = \
            -tf.reduce_sum(self.x * tf.log(1e-10 + self.x_reconstr_mean)
                           + (1 - self.x) * tf.log(1e-10 + 1 - self.x_reconstr_mean),
                           1)
        # 2.) The latent loss, which is defined as the Kullback Leibler divergence
        ##    between the distribution in latent space induced by the encoder on
        #     the data and some prior. This acts as a kind of regularizer.
        #     This can be interpreted as the number of "nats" required
        #     for transmitting the the latent space distribution given
        #     the prior.
        latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                           - tf.square(self.z_mean)
                                           - tf.exp(self.z_log_sigma_sq), 1)
        self.cost = tf.reduce_mean(reconstr_loss + latent_loss)  # average over batch
        # Use ADAM optimizer
        self.optimizer = \
            tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def partial_fit(self, X):
        """Train model based on mini-batch of input data.

        Return cost of mini-batch.
        """
        opt, cost = self.sess.run((self.optimizer, self.cost),
                                  feed_dict={self.x: X})
        return cost

    def transform(self, X):
        """Transform data by mapping it into the latent space."""
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.z_mean, feed_dict={self.x: X})

    def generate(self, z_mu=None):
        """ Generate data by sampling from latent space.

        If z_mu is not None, data for this point in latent space is
        generated. Otherwise, z_mu is drawn from prior in latent
        space.
        """
        if z_mu is None:
            z_mu = np.random.normal(size=self.network_architecture["n_z"])
        # Note: This maps to mean of distribution, we could alternatively
        # sample from Gaussian distribution
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.z: z_mu})

    def reconstruct(self, X):
        """ Use VAE to reconstruct given data. """
        return self.sess.run(self.x_reconstr_mean,
                             feed_dict={self.x: X})
def train_mnist(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5,num_layers=2):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size,
                                 num_layers = num_layers)
    #i put this in here so it will still work, but it won't be loaded when not using mnist
    import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    n_samples = mnist.train.num_examples

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, _ = mnist.train.next_batch(batch_size)

            # Fit training using batch data
            cost = vae.partial_fit(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae

def train(network_architecture, learning_rate=0.001,
          batch_size=100, training_epochs=10, display_step=5, num_layers=2,file_prefix='no_file.out'):
    vae = VariationalAutoencoder(network_architecture,
                                 learning_rate=learning_rate,
                                 batch_size=batch_size,
                                 num_layers=num_layers)
    #n_samples=1000
    # Training cycle
    #batch_xs is a matrix that is batch_size by len of each example
    waveform_data=dict()
    for t in [0, 1, 2,3]:
        waveform_data[t] = pickle.load(open(file_prefix +repr(t)+'.out', 'rb'))


    for epoch in range(training_epochs):
        #for m in [0]:#-1.0, -0.5, 0.0, 0.5, 1.0
        #this is for different noise means, off right now
        m=0.0
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            #batch_xs, _ = mnist.train.next_batch(batch_size)
            for tt in [0,1,2,3]:
                #have to put the signal between 0-1
                max_val_signal = waveform_data[tt][2]
                #max_val_fft = waveform_data[tt][3]
                batch_xs=get_waveform_batch(waveform_data[tt][0],i*batch_size,batch_size,m)
                #batch_xs = get_waveform_fft_batch(waveform_data[1], i * batch_size, batch_size, m)
                # Fit training using batch data
                #abs_batch=np.abs(batch_xs)
                batch_xs_rescale = (batch_xs + max_val_signal) / (max_val_signal * 2)
                #batch_xs_rescale=(batch_xs+max_val_fft) / (max_val_fft*2)
                cost = vae.partial_fit(batch_xs_rescale)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(avg_cost))
    return vae




########
#this was a bunch of test code
#I moved the part that I want to use to "train_and_test_vae.py"


#vae_20_fft=vae_2
#pickle.dump(vae_2,open('/home/new_sjohnston/vae_20_fft_2048_300_200_20_epochs.out','wb'))


# def recreate_time_series(fft_data):
#     #fft data is a vec with real part first then the imag part
#     #unstack add it back together then take conjugate and that back
#     #so that you can take ifft and get the time series back, then do spectro
#     #fft_data=fft_data[0,]
#     real_data=fft_data[range(int(len(fft_data)/2))]
#     imag_data = fft_data[np.array(range(int(len(fft_data) / 2)))+int(len(fft_data) / 2)]
#     c_data=real_data+1j*imag_data
#     #c_data_long=np.hstack((c_data,np.conj(c_data[len(c_data)::-1])))
#     return np.real(np.fft.ifft(c_data))

# #vae_30=vae_2
# test_case=13
# col_count=0
# plt.figure()
# for b in [1,2,3,4,5]:
#
#     col_count = col_count+1
#     for t in [0,1,2,3]:
#         if t == 0:
#             waveform_data = pickle.load(open('/home/new_sjohnston/waveform_data_fs_10000_100_data_points_nonoverlapping_low_noise_channel_0.out','rb'))
#             max_val_signal = waveform_data[2]
#             max_val_fft = waveform_data[3]
#             lowcut = 400
#             highcut = 5e3
#         elif t == 1:
#             waveform_data = pickle.load(open('/home/new_sjohnston/waveform_data_fs_10000_100_data_points_nonoverlapping_low_noise_channel_1.out','rb'))
#             max_val_signal = waveform_data[2]
#             max_val_fft = waveform_data[3]
#             lowcut = 400
#             highcut = 5e3
#         elif t == 2:
#             waveform_data = pickle.load(open('/home/new_sjohnston/waveform_data_fs_10000_100_data_points_nonoverlapping_low_noise_channel_2.out','rb'))
#             max_val_signal = waveform_data[2]
#             max_val_fft = waveform_data[3]
#             lowcut = 400
#             highcut = 5e3
#         elif t == 3:
#             waveform_data = pickle.load(open('/home/new_sjohnston/waveform_data_fs_10000_100_data_points_nonoverlapping_low_noise_channel_3.out','rb'))
#             max_val_signal = waveform_data[2]
#             max_val_fft = waveform_data[3]
#             lowcut = 400
#             highcut = 5e3
#
#         batch_xs_test=get_waveform_batch(waveform_data[0],0,100,0.0)
#         #batch_xs_test_1 = get_waveform_fft_batch(waveform_data[1], 57 * 1, 1, -1.0)
#
#         # batch_xs_test_1_real = batch_xs_test_1[range(int(len(batch_xs_test_1)/2))]
#         # batch_xs_test_1_imag = batch_xs_test_1[np.array(range(int(len(batch_xs_test_1) / 2)))+int(len(batch_xs_test_1) / 2)]
#         # batch_xs_test_1_comp = batch_xs_test_1_real + 1j*batch_xs_test_1_imag
#         # #batch_xs_rescale_1=batch_xs_test_1 * (max_val_signal*2)-max_val_signal
#         #x_reconstruct_11 = vae_2.reconstruct( (batch_xs_test_1 + max_val_signal) / (max_val_signal * 2))
#         #x_reconstruct_signal = vae_low_noise_20_300_200_100.reconstruct( (batch_xs_test_1 + max_val_signal) / (max_val_signal * 2))
#         vae_codes = vae_low_noise_2_300_200_100.transform(batch_xs_test)
#         plt.figure()
#         for i in range(2):
#             # plt.subplot(20,1,i+1)
#             plt.plot(vae_codes[:, i])
        # print('cw')
        # print(sum(vae_codes[:,cw]>3)/100)
        # print('down')
        # print(sum(np.logical_and((vae_codes[:,up[0]] < -3) ,(vae_codes[:,up[1]] < -3))) / 100)
        # print('up')
        # print(sum(np.logical_and((vae_codes[:,up[1]] > 3) , (vae_codes[:,up[0]] < -3))) / 100)

        # batch_xs_test_1 = batch_xs_test[1,]
        # # x_reconstruct_signal_scaled = x_reconstruct_signal * (max_val_signal * 2) - max_val_signal
        # # # x_reconstruct_11 = recreate_time_series(vae_2.reconstruct((batch_xs_test_1 + max_val_fft) / (max_val_fft * 2))*(max_val_signal*2)-max_val_signal-1)
        # # #batch_xs_test_1_time = recreate_time_series(batch_xs_test_1)
        # # #x_reconstruct_11_time = recreate_time_series(x_reconstruct_11[0,]*(max_val_signal*2)-max_val_signal-1)
        # spectro2=spectrogram(batch_xs_test_1, lowcut, highcut, 10000, int_time=128, novrlp=64)
        # # spectro2_reconstruct=spectrogram(x_reconstruct_signal_scaled[0,], lowcut, highcut, 10000, int_time=128, novrlp=64)
        # # #spectro2_reconstruct=spectrogram(x_reconstruct_11[0,], lowcut, highcut, 10000, int_time=128, novrlp=64)
        #
        # # plt.figure()
        # #
        # # plt.plot(x_reconstruct_11[0,]*(max_val_signal*2)-max_val_signal-1)
        # #
        # # plt.figure()
        # #
        # # plt.plot(batch_xs_rescale_1[0,])
        #
        # plt.figure()
        # plt.imshow(spectro2)

        # plt.subplot(4,5,t*5+col_count)
        # plt.imshow(spectro2_reconstruct)