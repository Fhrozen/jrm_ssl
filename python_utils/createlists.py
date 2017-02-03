import glob
import h5py
import os
import numpy as np
from random import shuffle

def create_lists(ang_info, tipos, proc, dir_cn, dir_ns):
    """ang_info = angle of the data
    tipos       = type of the data
    proc        = if is training or testing files
    dir_cn      = directory of clean data
    dir_ns      = directory of noisy data"""

    if ang_info == 'all':
        rango = np.arange(36)
    else:
        rango = [ang_info]
    if tipos == 'all': 
        tp_rg = ['noise', 'music']
    else:
        tp_rg = [tipos]
    if proc == 'trn':
        deep = '/**/**/**'
    else: 
        deep = '/**'
    cflist = []
    nflist = [[] for x in range(8)]
    all_tmp = []
    """ Create List """
    print 'Preparing {} data for {}:'.format(proc, tipos)
    for tipo in tp_rg:     
        for cnt in rango:
            indata_n = '{3}/{0}_{1}/D{2:03d}'.format(proc,tipo,cnt*10,dir_ns)
            tmp = glob.glob(indata_n + deep + '/*ch00.wav')
            all_tmp += tmp
    shuffle(all_tmp)
    cflist_tmp = all_tmp 
    for tipo in tp_rg:     
        for cnt in rango:
            indata_n = '{3}/{0}_{1}/D{2:03d}'.format(proc,tipo,cnt*10,dir_ns)
            cflist_tmp = [item.replace(indata_n,dir_cn) for item in cflist_tmp]
    cflist += [item.replace('_ch00','') for item in cflist_tmp]
    for k in range(8):
        arg = '_ch{0:02d}'.format(k)
        nflist_tmp = [item.replace('_ch00',arg) for item in all_tmp]
        nflist[k] += nflist_tmp
    lista = 'lista_{}.h5'.format(proc)
    if os.path.exists(lista):
        os.remove(lista)
    listfiles = h5py.File(lista,'a')
    dset = listfiles.create_dataset("cflist",data = cflist)
    dset = listfiles.create_dataset("nflist",data = nflist)
    listfiles.close()
    return nflist, cflist