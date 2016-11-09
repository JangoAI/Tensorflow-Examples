#coding=utf-8
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import codecs
import math
import argparse

IMAGES_DIR = "/home/guoqingpei/Desktop/test/data_NS100"

CWD = os.getcwd()
np.random.seed(1)


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log_dir','./log','directory for storing data')
flags.DEFINE_string('model_dir','./model','directory for storing model')
flags.DEFINE_integer('max_steps',10000,'directory for storing model')
flags.DEFINE_float('learning_rate',0.001,'learning rate for training')
flags.DEFINE_float('momentum',0.9,'momentum for training')
flags.DEFINE_bool('retrain',True,'completely retrain a new model,delete old models')
flags.DEFINE_integer('batch_size',50,'training or validate batch size of samples')



class Labels(object):
	'''
	definition for facial emotion labels
	include: copnversion from labels' num to labels' name and vice versa
	'''
	NE = 0
	AN = 1
	DI = 2
	HA = 3
	SA = 4
	SU = 5
	FE = 6

	@classmethod
	def getNumFromLabel(cls,label_name):
		#convert emotion label_name into number
		try:
			return cls.__dict__[label_name]
		except KeyError:
			return None

	@classmethod
	def getLabelFromNum(cls,num):
		#convert num into emotion label
		for k,v in cls.__dict__.items():
			if v==num:
				return k
		return None

	@classmethod
	def to_one_hot(cls,data):
		rows, cols = len(data),7
		one_hot_data = np.zeros([rows,cols])
		for row in range(rows):
			index = data[row]
			one_hot_data[row,index] = 1
		return one_hot_data




class DataSet(object):

	def __init__(self,real_data=True):
		self._images, self._labels = self.load_image(fake=not real_data)
		self._index_in_epoch = 0
		self._epoch_finished = 0
		self.samples = self._images.shape[0]

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def epoch_finished(self):
		return self._epoch_finished

	@property
	def index_in_epoch(self):
		return self._index_in_epoch

	def load_image(self, grayScaleNormalize=True,fake=True):
		if fake:
			print "loading fake data"
			images = np.random.random_sample([1000,64,64,1])
			labels = np.random.random_integers(0,6,size=(1000,))
		else:
			print "loading real data"
			filelists = codecs.open('img_idx','r','utf-8').readlines()
			filenames = [os.path.join(IMAGES_DIR,f.strip()) for f in filelists]
			labels = [Labels.getNumFromLabel(filename.split('_')[1]) for filename in filelists]
			images = []
			for imgfile in filenames:
				image = Image.open(imgfile, 'r').convert("L")
				image = np.asarray(image)
				if grayScaleNormalize:
					image = (image-np.mean(image))/np.std(image)
				image = image.reshape((64,64,1))
				images.append(image)
		assert len(images)==len(labels)
		return np.asarray(images),np.asarray(labels)

	def next_batch(self, batch_size=50,one_hot_label=False):
		start_index = self._index_in_epoch
		end_index = start_index + batch_size
		if end_index > self.samples:
			#end one epoch ,try to shuffle the images and labels
			reorder = np.arange(self.samples)
			np.random.shuffle(reorder)
			self._images = self._images[reorder]
			self._labels = self._labels[reorder]
			#reset start_index
			start_index = 0
			end_index = start_index + batch_size
			self._epoch_finished+=1
		self._index_in_epoch = end_index
		sl = slice(start_index, start_index+batch_size)
		return self._images[sl], (Labels.to_one_hot(self._labels[sl]) if one_hot_label else self._labels[sl])



def add_scalar_summary(input_tensor):
	#add summary
	mean_w = tf.reduce_mean(input_tensor,name=input_tensor.op.name+"/mean")
	min_w = tf.reduce_min(input_tensor,name=input_tensor.op.name+"/min")
	max_w = tf.reduce_max(input_tensor,name=input_tensor.op.name+"/max")
	tf.scalar_summary(mean_w.op.name,mean_w)
	tf.scalar_summary(min_w.op.name,min_w)
	tf.scalar_summary(max_w.op.name,max_w)
	tf.histogram_summary(input_tensor.op.name,input_tensor)




def weights(shape,xavier_params=(None,None),name="weights"):#use Xavier initialize
	fan_in, fan_out = xavier_params
	# low = -4*np.sqrt(12.0/(fan_in))
	# high = -low
	# init = tf.random_uniform(shape,minval=low,maxval=high,dtype=tf.float32)
	init = tf.truncated_normal(shape=shape, stddev=0.08)
	# init = np.random.randn(*shape)*math.sqrt(2.0/fan_in)
	W = tf.Variable(init,name=name,dtype=tf.float32)
	#add summary
	add_scalar_summary(W)
	return W



def bias(shape,name='bias'):
	# init = tf.random_uniform([shape],minval=-0.1,maxval=0.1)
	init = tf.constant(0.0,shape = shape)
	b =  tf.Variable(init,name=name,dtype=tf.float32)
	#add summary
	add_scalar_summary(b)
	return b



def conv(x,w):
	return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")




#will be referenced in both inference function and loss function
def placeholders():
	with tf.variable_scope("model_inputs"):
		images = tf.placeholder(dtype=tf.float32,shape=[None,64,64,1],name="input_image_batch")
		labels = tf.placeholder(dtype=tf.int32,shape=[None,],name="input_label_batch")
		keep_prob = tf.placeholder(tf.float32,name="keep_prob")
	return images, labels, keep_prob


def inference(images,dropout_keep_prob):
	"""Build the FER model up to where it may be used for inference."""
	#conv1
	with tf.variable_scope('conv1'):
		w = weights((5,5,1,32),(64*64*1,64*64*32))
		b = bias([32])
		conv1 = conv(images,w)+b#shape:64*64*32
		conv1_relu_out = tf.nn.relu(conv1)
		pooled_conv1_relu_out = tf.nn.max_pool(conv1_relu_out,[1,3,3,1],[1,2,2,1],padding='SAME')
		#tensor shape can be printed directly
		#shape 32*32*32
		print pooled_conv1_relu_out.get_shape()
		# print sess.run(pooled_conv1_relu_out.get_shape(),feed_dict={images:fake_data})
	with tf.variable_scope('conv2'):
		w = weights((5,5,32,32),(32*32*32,32*32*32))
		b = bias([32])
		conv2 = conv(pooled_conv1_relu_out,w)+b
		conv2_relu_out = tf.nn.relu(conv2)
		pooled_conv2_relu_out = tf.nn.max_pool(conv2_relu_out,[1,3,3,1],[1,2,2,1],padding='SAME')
		print pooled_conv2_relu_out.get_shape()
		#shape 16*16*32

	with tf.variable_scope('conv3'):
		w = weights((5,5,32,64),(16*16*32,16*16*64))
		b = bias([64])
		conv3 = conv(pooled_conv2_relu_out,w)+b
		conv3_relu_out = tf.nn.relu(conv3)
		pooled_conv3_relu_out = tf.nn.max_pool(conv3_relu_out,[1,3,3,1],[1,2,2,1],padding='SAME')
		print pooled_conv3_relu_out.get_shape()

	with tf.variable_scope("full_connnected_layer1"):
		flatten_out = tf.reshape(pooled_conv3_relu_out,(-1,8*8*64))
		w = weights((8*8*64,1000),(8*8*64,1000))
		b = bias([1000])
		f1 = tf.matmul(flatten_out,w)+b
		f1_relu_out = tf.nn.relu(f1)
		f1_relu_drop_out = tf.nn.dropout(f1_relu_out,dropout_keep_prob)
		print f1_relu_drop_out.get_shape()

	with tf.variable_scope("full_connnected_layer2"):
		w = weights((1000,1000),(1000,1000))
		b = bias([1000])
		f2 = tf.matmul(f1_relu_drop_out,w)+b
		f2_relu_out = tf.nn.relu(f2)
		f2_relu_drop_out = tf.nn.dropout(f2_relu_out,dropout_keep_prob)	
		print f2_relu_drop_out.get_shape()

	with tf.variable_scope("output_layer"):
		w = weights((1000,7),(1000,7))
		b = bias([7])
		logits = tf.matmul(f2_relu_drop_out,w)+b
		print logits.get_shape()

	return logits




def loss_function(logits, labels):
	"""Calculates the loss from the logits and the labels."""
	#Attention: logits must not be softmax regularized!!!!!!
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
		logits, labels, name='xentropy')
	loss = tf.reduce_sum(cross_entropy, name='xentropy_sum')
	return loss


def training_function(loss):
	# Add a scalar summary for the snapshot loss.
	tf.summary.scalar('loss', loss)
	# Create a variable to track the global step.
	global_step = tf.Variable(0, name='global_step', trainable=False)
	# Create the gradient descent optimizer with the given learning rate.
	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,1000,0.96,staircase=True)
	#summary for learning rate
	tf.summary.scalar('learning_rate', learning_rate)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	print global_step.op.name
	# Use the optimizer to apply the gradients that minimize the loss
	# (and also increment the global step counter) as a single training step.
	train_op = optimizer.minimize(loss, global_step=global_step)
	return train_op



def evaluate(sess, data,labels,images_placeholder,keep_prob_placeholder,labels_placeholder,
	eval_correct_op,logits,precision,batch_size=50):
	#pass whole data,every 100 images one time
	NUM_IMAGES = len(data)
	BATCH_NUM = NUM_IMAGES//batch_size
	correct_count = 0
	for i in range(BATCH_NUM):
		eval_data = data[i*batch_size:(i+1)*batch_size]
		eval_label = labels[i*batch_size:(i+1)*batch_size]
		correct_count+=sess.run(eval_correct_op,feed_dict={images_placeholder:eval_data,
			labels_placeholder:eval_label,
			keep_prob_placeholder:1.0})
	print "correct:%d / total:%d" % (correct_count,BATCH_NUM*batch_size)
	acc = correct_count/(BATCH_NUM*batch_size)
	#TODO:change acc to tensor
	precison_assign_op = tf.assign(precision,acc)
	sess.run(precison_assign_op)
	return acc



def run_training():
	g = tf.get_default_graph()
	with g.as_default():
		images, labels ,keep_prob = placeholders()
		sess = tf.Session()
		#output logits
		logits = inference(images, keep_prob)
		
		#get loss
		los = loss_function(logits, labels)
		
		#use loss value update model
		train_op = training_function(los)

		#calculate precision
		correct_flags = tf.cast(tf.nn.in_top_k(logits,labels,1),tf.float32)
		eval_correct_op = tf.reduce_sum(correct_flags)
		eval_correct_rate = tf.reduce_mean(correct_flags)
		precision = tf.Variable(0,name='precison',dtype=tf.float32,trainable=False)
		tf.scalar_summary(precision.op.name,precision)

		#collect summary
		summary = tf.summary.merge_all()
		
		# Instantiate a SummaryWriter to output summaries and the Graph.
		summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
		
		# Create a saver for writing training checkpoints.
		saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
		
		#steps to continue
		step = None
		
		#reload model
		if tf.gfile.Exists(FLAGS.model_dir):
			model_checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
			if model_checkpoint_path:
				saver.restore(sess,model_checkpoint_path)
				reader = tf.train.NewCheckpointReader(model_checkpoint_path)
				step =  reader.get_tensor("global_step")
				#get tensors' value in check point
				#print reader.get_tensor("conv1/bias")
				print "Model reload, Continue training..."
			else:
				#Attention here:must initialize all variables!
				sess.run(tf.initialize_all_variables())
				step = 0
		else:
			tf.gfile.MakeDirs(FLAGS.model_dir)
			sess.run(tf.initialize_all_variables())
			step = 0

		#training data ready
		dataset = DataSet(False)
		print "step:",step

		while step < FLAGS.max_steps:
			imgs, ls = dataset.next_batch(FLAGS.batch_size)
			feed_dict={images:imgs,labels:ls,keep_prob:0.5}
			_, losVal = sess.run([train_op,los],feed_dict=feed_dict)
			feed_dict={images:imgs,labels:ls,keep_prob:1.0}
			batch_acc= sess.run(eval_correct_rate,feed_dict=feed_dict)
			print 'Epoch:%s step:%s lossVal:%s batch_acc:%3f%%' % (dataset.epoch_finished,step,losVal,batch_acc*100) 
			if (step+1)%50==0:
				# Update the events file.
				feed_dict={images:imgs,labels:ls,keep_prob:1.0}
				summary_str = sess.run(summary, feed_dict=feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()
				print 'Collect Summary'
			if (step + 1) % 100 == 0 or (step + 1) == FLAGS.max_steps:
				#save model
				checkpoint_file = os.path.join(FLAGS.model_dir, 'model.ckpt')
				saver.save(sess, checkpoint_file, global_step=step)
				print 'Model Saved'
				print 'Test on whole dataset'
				#evaluate model on whole training and testing data
				total_acc = evaluate(sess,dataset.images,dataset.labels,images,
					keep_prob,labels,eval_correct_op,logits,precision,FLAGS.batch_size)
				# precison = sess.run(precision,feed_dict={labels:dataset.labels})
				print 'Epoch:%s step:%s lossVal:%s precison:%.3f%%' % (dataset.epoch_finished,step,losVal,total_acc*100)
			step+=1


def main(_):
	#clear log_dir
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
	tf.gfile.MakeDirs(FLAGS.log_dir)
	#retrain model every time!
	if FLAGS.retrain:
		if tf.gfile.Exists(FLAGS.model_dir):
			tf.gfile.DeleteRecursively(FLAGS.model_dir)
	run_training()


if __name__=="__main__":
	tf.app.run(main=main)
