# vae_signal
applying VAEs to detect and classify signals

if you run the "train_and_test_vae.py "it will create a set of files called 

python3 train_and_test_vae.py

that will generate a training set of 10000 signals for each type: upsweep, downsweep, tone, and noise.
then it will generate a test set of 100 each run through random channels

it will then plot the latent representation, just two digits, for the test signals

you can see that there is very little overlap between the vectors. I tried several different size for the latent representation but it kept using only two, which makes sense given the signals I'm using. Presumably they are representing center frequency and slope.

Obviously this is an easy case since the waveforms are completely distinct in frequency

But it is just a starting place.






