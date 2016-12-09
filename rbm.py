#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt



flags = tf.app.flags
FLAGS = flags.FLAGS

CWD = os.getcwd()

print "CWD:",CWD

flags.DEFINE_string('log_dir','./log','directory for storing data')
flags.DEFINE_string('model_dir','./model','directory for storing model')
flags.DEFINE_bool('retrain',False,'completely retrain a new model,delete old models')



#function to load mnist data
def load_mnist_dataset(mode='supervised', one_hot=True):
	"""Load the MNIST handwritten digits dataset.

	:param mode: 'supervised' or 'unsupervised' mode
	:param one_hot: whether to get one hot encoded labels
	:return: train, validation, test data:
	        for (X, y) if 'supervised',
	        for (X) if 'unsupervised'
	"""
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=one_hot)

	# Training set
	trX = mnist.train.images
	trY = mnist.train.labels

	# Validation set
	vlX = mnist.validation.images
	vlY = mnist.validation.labels

	# Test set
	teX = mnist.test.images
	teY = mnist.test.labels

	return trX, trY, vlX, vlY, teX, teY





class RBM(object):

	def __init__(self,
		batch_size = 100,
		feature_size = 784,
		learning_rate = 0.0001,
		gibbs_sample_steps = 3,
		epoch = 50,
		hidden_size=200):

		self.batch_size = batch_size
		self.feature_size = feature_size #shape:[None,feature_size]
		self.hidden_size = hidden_size
		self.gibbs_sample_steps = gibbs_sample_steps
		self.learning_rate = learning_rate
		self.epoch = epoch

		self.inference()

		self.sess = tf.Session()

		# Create a saver for writing training checkpoints.
		self._saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

		self.checkpoint_file = os.path.join(FLAGS.model_dir, 'model.ckpt')
	
	def save_model(self):
		print "model saved:%s" % self.checkpoint_file
		self._saver.save(self.sess, self.checkpoint_file)

	def inference(self):
		#placeholders
		self.input = tf.placeholder(dtype=tf.float32,shape=[self.batch_size ,self.feature_size])
		self.random_h_val = tf.placeholder(dtype=tf.float32,shape=[self.batch_size ,self.hidden_size])

		#Variables
		self.W = tf.Variable(tf.random_normal([self.feature_size, self.hidden_size],
			stddev=0.1,name='weights'))
		self.v_b = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[self.feature_size,]),
			name="v_b")
		self.h_b = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[self.hidden_size,]),
			name="h_b")
		
		#logic
		self.encode_pro = self._infer_h_from_v(self.input)[0]
		self.reconstruct_pro = self._infer_v_from_h(self.encode_pro)


		pro_h, sample_h, pro_v, pro_h2, sample_h2 = self.gibbs_sample_step(self.input)

		#self.input and pro_h ~ p(h/v)
		positive_w = tf.matmul(tf.transpose(self.input), pro_h)
		positive_hb = pro_h
		positive_vb = self.input

		#using gibbs sampling to get (v,h) ~ p(v,h)
		nn_input = pro_v
		for _ in range(self.gibbs_sample_steps):
			pro_h, sample_h, pro_v, pro_h2, sample_h2 = self.gibbs_sample_step(nn_input)
			nn_input = pro_v

		#pro_v, and pro_h2 ~ p(v,h)
		negative_w = tf.matmul(tf.transpose(pro_v), pro_h2)
		negative_hb = pro_h2
		negative_vb = pro_v

		self.update_W_op = tf.assign_add(self.W, self.learning_rate*(positive_w-negative_w)/self.batch_size)
		self.update_hb_op = tf.assign_add(self.h_b, self.learning_rate*tf.reduce_mean(positive_hb-negative_hb,0))
		self.update_vb_op = tf.assign_add(self.v_b, self.learning_rate*tf.reduce_mean(positive_vb-negative_vb,0))
		
		#loss function
		self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.input, pro_v))))

		#collect summary
		# summary = tf.summary.merge_all()
		# Instantiate a SummaryWriter to output summaries and the Graph.
		# summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, self.sess.graph)
		# Create a saver for writing training checkpoints.
		# saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)


	def reconstruct(self,data):
		model_checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)	
		if model_checkpoint_path:
			self._saver.restore(self.sess,model_checkpoint_path)
			print "load recent model"
		feed_dict = self._create_feed_dict(data)
		recon = self.sess.run(self.reconstruct_pro,feed_dict=feed_dict)
		return recon

	def train(self,restore=True):
		trX, trY, vlX, vlY, teX, teY = load_mnist_dataset()
		self.sess.run(tf.initialize_all_variables())
		if restore:
			model_checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)	
			if model_checkpoint_path:
				self._saver.restore(self.sess,model_checkpoint_path)
				print "restore prior trained model:%s" % model_checkpoint_path
		for i in range(self.epoch):
			print "Epoch: %d" % i
			self.train_one_epoch(trX)
			self.save_model()


	def train_one_epoch(self,data):
		data = np.array(data)
		for i in range(0,data.shape[0],self.batch_size):
			batch = data[i:i+self.batch_size]
			feed_dict = self._create_feed_dict(batch)
			updates = [self.update_W_op ,self.update_hb_op, self.update_vb_op]
			self.sess.run(updates,feed_dict=feed_dict)
			#loss
			los = self.sess.run(self.loss,feed_dict=feed_dict)
		print("loss: %f" % los)

	def _create_feed_dict(self,data):
		feed_dict = {self.input:data,
					self.random_h_val:np.random.rand(data.shape[0], self.hidden_size),}

		return feed_dict

	def gibbs_sample_step(self,v):
		#sample from v_pro
		pro_h,sample_h = self._infer_h_from_v(v)
		pro_v = self._infer_v_from_h(pro_h)
		pro_h2, sample_h2 = self._infer_h_from_v(pro_v)
		return pro_h, sample_h, pro_v, pro_h2, sample_h2

	def _infer_h_from_v(self,v_sample):
		#v_shape = [self.batch_size,feature_size]
		pro_h = tf.sigmoid(tf.matmul(v_sample,self.W)+self.h_b)
		#get random numbers range from [0,1)
		sample_h = tf.nn.relu(tf.sign(pro_h-self.random_h_val))
		return pro_h, sample_h

	def _infer_v_from_h(self,pro_h):
		pro_v = tf.sigmoid(
			tf.matmul(pro_h, tf.transpose(self.W))+self.v_b)
		return pro_v



def main(_):
	#clear log_dir
	if tf.gfile.Exists(FLAGS.log_dir):
		tf.gfile.DeleteRecursively(FLAGS.log_dir)
		tf.gfile.MakeDirs(FLAGS.log_dir)
	#retrain model every time!
	if tf.gfile.Exists(FLAGS.model_dir):
		if FLAGS.retrain:
			tf.gfile.DeleteRecursively(FLAGS.model_dir)
	else:
		tf.gfile.MakeDirs(FLAGS.model_dir)
	rbm = RBM(epoch=500)
	# rbm.train()
	trX, trY, vlX, vlY, teX, teY = load_mnist_dataset()
	ori = vlX[:100]
	recon = rbm.reconstruct(ori)
	ori = ori.reshape([-1,28,28])
	recon  = recon.reshape([-1,28,28])
	figs, axes = plt.subplots(10, 2, figsize=(7,7))
	for ax in axes.flatten():
		ax.set_xticks([])
		ax.set_yticks([])
		ax.axis('off')
	for i in range(10):
		index = np.random.random_integers(0,99)
		axes[i,0].imshow(ori[index],cmap='gray')
		axes[i,1].imshow(recon[index],cmap='gray')
	plt.show()




if __name__=="__main__":
	tf.app.run(main=main)
