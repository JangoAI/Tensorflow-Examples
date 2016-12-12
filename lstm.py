#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist = input_data.read_data_sets("./MNIST_data", one_hot=False)

num_units = 250

feature_length = 28
time_steps = 28
batch_size = 200


#placeholder
input_data = tf.placeholder(tf.float32,shape=[None,time_steps,feature_length])
input_label = tf.placeholder(tf.int64,shape=[None])

#variable
w = tf.get_variable("W",[2*num_units,10])
b = tf.get_variable("b",[10])

#preprare input
x = tf.transpose(input_data,[1,0,2])#(time_steps,batch_size,feature_length)
x = tf.reshape(x,[-1,feature_length])#(time_steps*batch_size,feature_length)
x = tf.split(0,time_steps,x)

#build rnn 
# Forward direction cell
lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1.0)
# Backward direction cell
lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1.0)
lstm_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell]*3, state_is_tuple=True)
lstm_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell]*3, state_is_tuple=True)
outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw, lstm_bw, x,dtype=tf.float32)
#single direction LSTM
# rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units, state_is_tuple=True)
# lstm = tf.nn.rnn_cell.MultiRNNCell([rnn_cell]*3, state_is_tuple=True)
# outputs, state = tf.nn.rnn(lstm,x,dtype=tf.float32)
# print outputs
logits = tf.matmul(outputs[-1],w)+b

#loss
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, input_label))
#precison
prec = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits,1),input_label),tf.float32))
#train
train_op = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	for i in range(100):
		batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape([batch_size,time_steps,feature_length])
		_, los, pre = sess.run([train_op,loss,prec],feed_dict={input_data:batch_x,input_label:batch_y})
		print "los:%3f precision:%3f" %(los, pre)

	# Calculate accuracy for 128 mnist test images
	test_len = 128
	test_data = mnist.test.images[:test_len].reshape((-1, time_steps, feature_length))
	test_label = mnist.test.labels[:test_len]
	print("Testing Accuracy:", \
		sess.run(prec, feed_dict={input_data: test_data, input_label: test_label}))
