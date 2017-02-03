#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import signal, timeit, ConfigParser, imp, importlib, argparse
import sys, os, h5py, six, threading
from time import localtime, strftime, sleep
import numpy as np
from datetime import datetime
from six.moves import queue
from random import shuffle
from sys import stdout 
from multiprocessing import Pool

import chainer
import model_chainer
from chainer import computational_graph as Graph

parser = argparse.ArgumentParser(description='Training Program based on Chainer Framework')
parser.add_argument('A', metavar='action', type=str, help='Specify an option. t: train, r:resume, f:finetune')
parser.add_argument('-C', metavar='config_file', type=str, default='', help='String of the Config File')
parser.add_argument('-R', metavar='resume_file', type=str, default='', help='String with the date of the resume training')
parser.add_argument('-F', metavar='fintune_file', type=str, default='', help='String with the date of the finetune training')
args = parser.parse_args()
config = ConfigParser.ConfigParser()

def set_environment(Option):
    global init_epoch, batch_init
    global epochs, batchsize
    global start, finetune
    global config
    global folders_list
    global Model
    global logger 

    init_epoch = 1
    batch_init = 0
    start = True
    current_exp = strftime("%Y-%m-%d_%H%M", localtime())
    logger = []
    if Option == 't':
        print('Setting the environment for training...')
        print('  the files will be saved at: {}'.format(current_exp))
        config.readfp(open(args.C))
        config.read(args.C)
        app_folder = os.path.dirname(os.path.abspath(args.C))
        app_folder = '{}/training_files/{}'.format(app_folder, current_exp)
        folders_list = ['', '/train', '/graph', '/logs'] 
        folders_list = ['{}{}'.format(app_folder, x) for x in folders_list]
        for folder in folders_list:
            os.makedirs(folder)        
        net = imp.load_source('Network', config.get('network', 'file'))   
        with open(config.get('network', 'file'), 'r') as f:
            net_def=f.read()   
        with open('{}/network.py'.format(folders_list[0]), 'w+') as f: 
            f.write(net_def)                                           
        with open(args.C, 'r') as f:
            cfg = f.read()
        with open('{}/train_01.cfg'.format(folders_list[0]), 'w+') as f:
            f.write(cfg)
        epochs = config.getint('train', 'epochs')
        batchsize = config.getint('train', 'batch')
        print('  epochs:    {}'.format(epochs))
        print('  batchsize: {}'.format(batchsize))
    elif Option == 'r':
        print('Setting the environment to resume the training...')
        config.readfp(open(args.R))
        config.read(args.R)
        app_folder = os.path.dirname(os.path.abspath(args.R))
        folders_list = ['', '/train', '/graph', '/logs'] #
        folders_list = ['{}{}'.format(app_folder, x) for x in folders_list]
        logg_name = '{}/logg.h5'.format(folders_list[3]) 
        with h5py.File(logg_name, 'r') as f:
            logger = np.zeros(f['logger'].shape)
            np.copyto(logger, f['logger'])
        logger = logger.tolist()
        net = imp.load_source('Network', '{}/network.py'.format(folders_list[0]))   
        init_epoch = int(logger[-1][1]) 
        batch_init = int(logger[-1][2]) 
        start = False
        epochs = config.getint('train', 'epochs')
        batchsize = config.getint('train', 'batch')
        print('  epochs:    {:03d}, init epoch: {}'.format(epochs, init_epoch))
        print('  batchsize: {:03d}, init batch: {}'.format(batchsize, batch_init))
    elif Option == 'f':
        print('Setting the environment for finetune the model...')
        print('  the files will be saved at: {}'.format(current_exp))
        config.readfp(open(args.F))
        config.read(args.F)
        app_folder = os.path.dirname(os.path.abspath(args.F))
        app_folder = '{}/training_files/{}'.format(app_folder, current_exp)
        folders_list = ['', '/train', '/graph', '/logs'] #
        folders_list = ['{}{}'.format(app_folder, x) for x in folders_list]
        for folder in folders_list:
            os.makedirs(folder)        
        net = imp.load_source('Network', config.get('network', 'file'))   
        with open(config.get('network', 'file'), 'r') as f:
            net_def=f.read()   
        with open('{}/network.py'.format(folders_list[0]), 'w+') as f: 
            f.write(net_def)                                           
        with open(args.F, 'r') as f:
            cfg = f.read()
        with open('{}/train_01.cfg'.format(folders_list[0]), 'w+') as f:
            f.write(cfg)
        epochs = config.getint('train', 'epochs')
        batchsize = config.getint('train', 'batch')
        finetune = []
        print('  epochs:    {}'.format(epochs))
        print('  batchsize: {}'.format(batchsize))
        finetune = [config.get('network', 'ft_prefix'), config.getint('network', 'ft_epoch'), config.getint('network', 'ft_batch')]
    else:
        raise ImportError('Unrecognized option, please select one: t (train), r (resume), f (finetune)')
    Model = model_chainer.Model(config, net.Network) 
    return

def savelog():
    logg = np.asarray(logger)
    logg_name = '{}/logg.h5'.format(folders_list[3])
    if os.path.exists(logg_name):
        os.remove(logg_name)
    with h5py.File(logg_name, 'a') as f:
         ds = f.create_dataset('logger', data=logg)   
    return

def data_feed():
    global max_len
    global batch_init
    num_networks = [int(x) for x in config.get('gpu', 'index').split(';')  ]
    num_networks = np.amax((len(num_networks),1)).astype(np.int)
    DBClass = importlib.import_module('python_utils.datareader.{}'.format(
        config.get('reader', 'data')))
    reader = getattr(DBClass,config.get('reader', 'class'))(config)
    idxs = reader.idxs
    max_len = len(idxs) - (len(idxs) % (batchsize*num_networks))
    data_q.put('train')
    if args.A == 'r': batch_init += batchsize
    for epoch in six.moves.range(init_epoch,1+epochs):
        shuffle(idxs)
        for idx in range (batch_init,max_len,batchsize*num_networks):
            data_batch = reader.read_data(idxs[idx:idx+batchsize], num_networks)
            data_q.put((epoch, idx, data_batch.copy()))
        batch_init = 0
    data_q.put('end')
    return

def log_result():
    #Logger
    global logger
    display_msg = config.getint('train', 'display_log')
    step  = 0
    temporizer = 0
    format_str = ('%s: epoch %.4f, loss = %.6f, acc = %.2f (%.1f examples/sec; %.3f sec/batch; %.2f mins to finish)')
    while True:
        result = res_q.get()
        if result == 'end':
            print('Optimization Finished')
            savelog()
            print('Files saved at: {}'.format(folders_list[0]))
            print('Training Process took {:.2f} mins'.format(logger[-1][0]))
            break
        elif result == 'train':
            train = True
            continue
        epoch, index_epoch, loss_value, acc, duration = result
        temporizer += (duration)/60.0
        if train:
            logger.append([temporizer,epoch,index_epoch,loss_value,acc])
            if (step % display_msg== 0):
                examples_per_sec = batchsize / float(duration)
                current_epoch = float(epoch) + float(index_epoch)/float(max_len)
                sec_per_batch = float(duration)
                waiting_time = (epochs + 1. - current_epoch ) * sec_per_batch * max_len / (60. * batchsize) 
                print(format_str % (strftime("%Y-%m-%d %H:%M:%S", localtime()), current_epoch, loss_value, acc,
                        examples_per_sec, sec_per_batch, waiting_time))
            step += 1
    return

def train():
    global Model
    global start
    Model.train = True
    if args.A == 'r':
        Model.LoadResumeModel(folders_list[1], init_epoch, batch_init)
    elif args.A == 'f': 
        Model.LoadFineTnModel(finetune[0], finetune[1], finetune[2])
    Model.SetUp()
    if args.A == 'r':
        Model.LoadTraining(folders_list[1], init_epoch, batch_init)
        print('Resumming Training')
    
    while True:
        t_start = timeit.default_timer()  
        while data_q.empty():
            sleep(0.1)
        inp = data_q.get()
        if inp == 'end':
            Model.train = False
            Model.save(folders_list[1], logger[-1][1], logger[-1][2])
            res_q.put('end')
            break
        elif inp == 'train':
            res_q.put('train')
            Model.train = True
            continue
        volatile = 'off' if Model.train else 'on'

        epoch, index_epoch, data = inp
        if Model.train:
            Model.ZeroGrads()
            Model(data)
            Model.BackWard()
            Model.UpDate()
            if start:
                start = False
                variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}
                function_style = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}
                g = Graph.build_computational_graph((Model.loss, ),
                    variable_style=variable_style, 
                    function_style=function_style)
                with open('{}/graph_model.dot'.format(folders_list[2]), 'w') as o:
                    o.write(g.dump())
                print('graph generated')
            tmp_loss = float(Model.loss.data) #
            if (Model.accuracy is None): 
                accuracy = 0
            else:
                accuracy = float(Model.accuracy.data)
            t_stop = timeit.default_timer()-t_start
            stdout.write('current loss={:.05f}, acc={:.03f}, time={:.4f}\r'.format(tmp_loss*batchsize, accuracy, t_stop))
            stdout.flush()
            res_q.put((epoch, index_epoch,tmp_loss,accuracy, t_stop))
            del data
    return

"""Definition CTRL+C Terminate"""
def signal_handler(signal, frame):
    print('\n...Previous Finish Training')
    Model.save(folders_list[1], logger[-1][1], logger[-1][2])
    savelog()
    print('Files saved at: {}'.format(folders_list[0]))
    print('Training Process took {:.2f} mins'.format(float(logger[-1][0])))
    print('Optimization Finished')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def main():
    feeder = threading.Thread(target = data_feed)
    feeder.daemon = True
    feeder.start()
    logger = threading.Thread(target=log_result)
    logger.daemon = True
    logger.start()

    train()
    feeder.join()
    logger.join()
    return

if __name__=='__main__':
    print('Training Program based on Chainer Framework')
    Model = None
    set_environment(args.A)
    data_q = queue.Queue(maxsize = 1)
    res_q = queue.Queue()
    main()
    sys.exit(0)
