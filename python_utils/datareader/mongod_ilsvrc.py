from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, ConfigParser
import tensorflow as tf
from pymongo import MongoClient
import gridfs
import numpy as np
from tempfile import TemporaryFile
from skimage.transform import resize

class tf_images(object):

	def __init__(self, config_file):
		_ip=config_file.get('server', 'ip')
		_port=config_file.get('server','port')
		self.server = 'mongodb://{}:{}'.format(_ip, _port)
		self.INDEXES = config_file.getint('server', 'index')
		self.DB_PREFIX = config_file.get('server', 'db_prefix')
		self.img_shape = config_file.getint('data', 'img_shape') 
		#TODO: Add Recursivity for multiple sizes layes
		self.input_ = config_file.get('data', 'input_label')
		self.target_ = config_file.get('data', 'target_label') 	
		#TODO: Add Recursivity for multiple layes
		self.get_db_sizes()	

	def get_db_sizes(self):
		_idxs = []
		Client = MongoClient(self.server, connect=False)
		print('Looking at databases:')
		for i in range(self.INDEXES):
			fs = gridfs.GridFS(Client['{}_{:02d}'.format(self.DB_PREFIX, i)])
			 #TODO:set globals number of index 
			fls = fs.list()
			for j in fls:
				_idxs.append('{}_{}'.format(i, int(j.strip('id'))))
		self.idxs = _idxs
		self.datalen = len(_idxs)
		print('  Total of {} files...'.format(len(self.idxs)))
		return

	def read_data(self, idxs):
		"""
			Data feeder from MongoDB Server
			idxs[,0] = Number of file to look for
			indx[,1] = Class of the file
		"""
		Client = MongoClient(self.server, connect=False)
		_idxs = idxs #In case of using tensors or smth
		batch_in = len(_idxs)
		input_data = np.zeros((batch_in,self.img_shape,self.img_shape,3), dtype=np.float32)
		target_data = np.zeros((batch_in,), dtype=np.int32)
		for i in xrange(batch_in):
			iDB, iFL = _idxs[i].split('_')
			db = Client['{}_{:02d}'.format(self.DB_PREFIX,int(iDB))]
			fs = gridfs.GridFS(db)
			flname = fs.find_one({'filename':'id{:05d}'.format(int(iFL))})
			tmpFl = TemporaryFile()
			tmpFl.write(flname.read())
			tmpFl.seek(0)
			np_fl = np.load(tmpFl)
			img = (resize(np_fl[self.input_], [self.img_shape, \
				self.img_shape]).astype(np.float32)) #Image Data
			# TODO: Add Mean Extraction and Normalization
			#input_data[i] = np.swapaxes(img, 0, 2)
			input_data[i] = img - 0.5
			target_data[i] = np_fl[self.target_] #Array Data
		#input_data = tf.Variable(input_data, trainable=False, collections=[]) 
		#target_data = tf.Variable(target_data, trainable=False, collections=[]) 
		return input_data, target_data
	
	def placeholders(self, capacity):
		x = tf.placeholder(tf.float32, [capacity, self.img_shape, self.img_shape, 3], name='x-input')
		y = tf.placeholder(tf.int32, [capacity], name='y-input')
		#shapes = [[capacity, self.img_shape, self.img_shape, 3],[capacity]]
		return x, y#, shapes

	def next_step(self, batchsize, min_after_dequeue, nepochs=None):
		capacity = min_after_dequeue + 3*batchsize
		filename_queue = tf.train.string_input_producer(
			self.idxs, num_epochs=nepochs, shuffle=True)
		#input_data, label_data = self.read_data(filename_queue)
		lista = tf.cast(filename_queue.dequeue_many(capacity), dtype='string')
		#tf.Print(lista.eval())
		
		"""
		input_batch, label_batch = tf.train.shuffle_batch(
			[input_data, label_data], batch_size=batchsize, capacity=capacity,
			min_after_dequeue=min_after_dequeue)
		return input_batch, label_batch
		"""
		return lista