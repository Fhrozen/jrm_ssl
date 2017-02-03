#!/usr/bin/python
import os, glob, h5py, ConfigParser
import numpy as np
from sklearn.metrics import confusion_matrix

config = ConfigParser.ConfigParser()
exp = 2
base_folder = sorted(glob.glob('./training_files/*'))[exp]
print('Runing results from folder {}'.format(base_folder))
config_file='{}/train_01.cfg'.format(base_folder)
config.readfp(open(config_file))
config.read(config_file)
print(config.get('general', 'description'))

data_folder = '{}/Results_MultiNoise'.format(base_folder)
snr_vals = ['Clean', 45, 30, 15, 10, 5, 0, -5,-10,-15,-20,-25,-30,-35]

for snr in snr_vals:
    _snr = str(snr).replace('-', 'm')
    files = glob.glob('{}/{}dB/**/**/*h5'.format(data_folder, _snr))
    predictd_angle = []
    true_angle = []
    mad_angle = []
    l_real =  'real' 
    l_predict = 'predict' 
    for archivo in files:
        with h5py.File(archivo, 'r') as f:
            _true_angle = np.zeros(f[l_real].shape, dtype = np.int32)
            np.copyto(_true_angle, f[l_real])
            outdata =  np.zeros(f[l_predict].shape, dtype = np.float32)
            np.copyto(outdata, f[l_predict])
        true_angle += [np.amin(_true_angle)]
        argmax_angle = np.argmax(outdata, axis=1)
        predictd_angle += [np.median(argmax_angle)]
        mad_angle += [np.median(np.absolute(argmax_angle - (predictd_angle[-1])*np.ones((argmax_angle.size))))]
    savefile = '{}/BlockAcc'.format(base_folder)
    if not os.path.exists(savefile):
        os.makedirs(savefile) 
    savefile = '{}/values_{}.h5'.format(savefile, _snr)
    predictd_angle = np.asarray(predictd_angle, dtype=np.float32)
    true_angle = np.asarray(true_angle, dtype=np.float32)
    mad_angle = np.asarray(mad_angle, dtype=np.float32)
    
    #Additional Results 
    _AV_MAD = np.mean(mad_angle)
    labels = range(0,365,5)
    _CM = confusion_matrix(true_angle.astype(np.int), predictd_angle.astype(np.int), labels=labels)
    _ACC = np.diagonal(_CM)
    _ACC = np.sum(_ACC[0:72].astype(np.float32))/(72*50)
    print('SNR: {} ACC: {:.04f} AV_MD: {:.02f}'.format(snr, _ACC, _AV_MAD))
    if os.path.exists(savefile):
        os.remove(savefile)
    with h5py.File(savefile,'a') as f:
        dset = f.create_dataset("predicted", data=predictd_angle)
        dset = f.create_dataset("true_angle", data=true_angle)
        dset = f.create_dataset("mad_angle", data=mad_angle)
        dset = f.create_dataset("avmad", data=_AV_MAD)
        dset = f.create_dataset("cm", data=_CM)
 
print('Finish')
