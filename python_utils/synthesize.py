import scipy.io.wavfile as wavfile
import numpy as np
from numpy import linalg as LA
from scikits import audiolab


def synth_audio(audiofile, impfile, chns, angle, nsfile=None, snrlevel=None, outname=None, outsplit=False):
	FreqSamp, audio = wavfile.read(audiofile) 
	audio = audio.astype(np.float32)/np.amax(np.absolute(audio.astype(np.float32)))
	gen_audio = np.zeros((audio.shape[0], chns), dtype=np.float32)
	for ch in range(1,chns+1):
		impulse = np.fromfile('{}D{:03d}_ch{}.flt'.format(impfile, angle, ch), dtype=np.float32)
		gen_audio[:,ch-1] = np.convolve(audio, impulse, mode='same')

	gen_audio = add_noise(gen_audio, nsfile=nsfile, snrlevel=snrlevel)

	if outname is None: 
		return FreqSamp, np.transpose(gen_audio)
	if outsplit:
		for ch in range(chns):
			play_data = audiolab.wavwrite(gen_audio[:,ch],'{}_ch{:02d}.wav'.format(outname,ch), fs=FreqSamp, enc='pcm16')
		return
	else:
		play_data = audiolab.wavwrite(gen_audio,'{}.wav'.format(outname), fs=FreqSamp, enc='pcm16')
	return

def add_noise(gen_audio, nsfile=None, snrlevel=None):
	chns = gen_audio.shape[1]

	if not ((nsfile is None) or (nsfile==-1)):
		_, noise= wavfile.read(nsfile) 
		noise = noise[0:gen_audio.shape[0]]

	if not (snrlevel is None or snrlevel=='Clean'):
		if nsfile is None:
			noise = np.random.uniform(-1.0, 1.0, (gen_audio.shape[0],)) 
		if nsfile == -1:
			noise = np.random.uniform(-1.0, 1.0, (gen_audio.shape[0], chns)) 
		else:
			noise = np.tile(noise[:,np.newaxis], [1, chns])
		noise = noise.astype(np.float32)/np.amax(np.absolute(noise.astype(np.float32)))
		noise = noise/LA.norm(noise) * LA.norm(gen_audio) / np.power(10,0.05*float(snrlevel))

		gen_audio= gen_audio+noise

	gen_audio /=np.amax(np.absolute(gen_audio))  #Normalized Audio

	return gen_audio



