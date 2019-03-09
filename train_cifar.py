import tensorflow as tf
import numpy as np
from cifar_setup import *
from model import Model
import random
import os
from cifar_setup import *
from cifar_setup2 import *

data_path = './data_set/cifar_10/'

raw_cifar = CIFAR10Data(data_path)

(x_train, y_train), (x_test, y_test) = get_data_set("train"), get_data_set("test")
#x_train = tf.reshape(x_train, [-1, 32, 32, 3])
n = y_train.shape[0]

model = Model()
saver = tf.train.Saver(max_to_keep=3)
model_dir = "./model"
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

global_step = tf.contrib.framework.get_or_create_global_step()

# lr =  learning_rate = tf.train.exponential_decay(1e-4, 50000,
#  100, 0.96, staircase=True)
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.loss)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
cifar = AugmentedCIFAR10Data(raw_cifar, sess, model)

epoch_num = 50000
batch_size = 256
num_checkpoints = 10000

for i in range(epoch_num):

    inds = random.sample(range(n), batch_size)
    x_batch, y_batch = x_train[inds, :], y_train[inds]
    # print(x_batch[1, np.arange(1,32)])
    # x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
    #                                                    multiple_passes=True)
    # print (x_batch[1, np.arange(1,32), 0, 0])
    # print (y_batch[np.arange(1,20)])
    nat_dict = {model.x_input: x_batch, model.y_input: y_batch}
    sess.run(train_step, feed_dict = nat_dict)
    if i % 100 == 0:
        inds_eval = random.sample(range(10000), batch_size)
        x_eval, y_eval = x_test[inds_eval, :], y_test[inds_eval]

        eval_dict = {model.x_input: x_eval, model.y_input: y_eval}
        print ("epoch: "+str(i * 256. / 10000.) + ", accuracy: " +
               str(sess.run(model.accuracy, feed_dict = eval_dict)))

    if i % num_checkpoints == 0:
        saver.save(sess,
                   os.path.join(model_dir, 'checkpoint'),
                   global_step=global_step)