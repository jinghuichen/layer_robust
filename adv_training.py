import tensorflow as tf
import numpy as np
from cifar_setup import *
from model import Model
import random
import os
from cifar_setup import *
from cifar10_input import *
from datetime import datetime
from timeit import default_timer as timer
import argparse

class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, ind, random_start):
        """Attack parameter initialization. The attack performs k steps of
             size a, while always staying within epsilon from the initial
             point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.ind = ind
        
        loss = model.loss
        noise_name = "noise" + str(ind)
        self.get_grad = tf.gradients(model.loss, self.model.noise[noise_name])[0]
     #   self.grad = tf.gradients(loss, model.x_input)[0]
#         self.grad1= tf.gradients(loss, model.noise1)[0]
#         self.grad2= tf.gradients(loss, model.noise2)[0]
#         self.grad3= tf.gradients(loss, model.noise3)[0]
#         self.grad4= tf.gradients(loss, model.noise4)[0]
#         self.grad5= tf.gradients(loss, model.noise5)[0]
#         self.grad6= tf.gradients(loss, model.noise6)[0]
#         self.grad7= tf.gradients(loss, model.noise7)[0]
         
 
    def get_perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
             examples within epsilon of x_nat in l_infinity norm."""
        x = x_nat
        
        #get all outputs
        outputs = {}
        outputs["h1"], outputs["h2"], outputs["h3"], outputs["h4"], outputs["h5"], outputs["h6"], outputs["h7"] = sess.run([self.model.relu1, self.model.pooling2, self.model.relu3, self.model.pooling4, self.model.relu5, self.model.pooling6, self.model.relu_dense], feed_dict={self.model.x_input: x, self.model.y_input: y})


        noise_name = "noise" + str(self.ind)
        output_name = "h" + str(self.ind)
        
        epsilon = self.epsilon
        step_size = self.step_size
     
        if self.ind == 0:
            noise = np.zeros(x.shape)
            noise_nat = np.copy(noise)
        else:
            noise = np.zeros(outputs[output_name].shape)
            noise_nat = np.copy(noise)
        
        grad = sess.run(self.get_grad, feed_dict={self.model.x_input: x, self.model.y_input: y, self.model.noise[noise_name]:noise})
        np.add(noise, step_size * np.sign(grad), out=noise, casting='unsafe')
        noise = np.clip(noise, noise_nat - epsilon, noise_nat + epsilon)

        #      print (noise_nat)
       # print (np.linalg.norm(outputs[output_name]))
        outputs.clear()
        
        
        return noise



parser = argparse.ArgumentParser(description='CIFAR10 adversarial training')    
parser.add_argument('--ind', default=1, type=int, help='noise ind')    
parser.add_argument('--eps', default=0.01, type=float, help='eps budget')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')

args = parser.parse_args()


data_path = './data_set/cifar_10/'
#
# if not os.path.exists('./data_set/cifar_10/batches.meta'):
#     maybe_download_and_extract()

(x_train, y_train), (x_test, y_test) = get_data_set("train"), get_data_set("test")
#x_train = tf.reshape(x_train, [-1, 32, 32, 3])
# raw_cifar = CIFAR10Data(data_path)
n = y_train.shape[0]

model_file = tf.train.latest_checkpoint('./model')
#model_file = './model_adv/checkpoint_0_eps_0.02_iter_10000-0'

model = Model()
ind_noise = args.ind
attack = LinfPGDAttack(model,
                           epsilon = args.eps,
                           num_steps = 20,
                           step_size = args.eps,
                           ind = ind_noise,
                           random_start = False)
saver = tf.train.Saver(max_to_keep=3)
model_dir = "./model_adv"



if not os.path.exists(model_dir):
    os.makedirs(model_dir)

global_step = tf.contrib.framework.get_or_create_global_step()

# lr =  learning_rate = tf.train.exponential_decay(1e-4, 50000,
#  100, 0.96, staircase=True)

# train_step = tf.train.AdamOptimizer(1e-4).minimize(model.loss)

# step_size_schedule = [[0, 0.01], [30000, 0.001], [40000, 0.0001]]
# boundaries = [int(sss[0]) for sss in step_size_schedule]
# boundaries = boundaries[1:]
# values = [sss[1] for sss in step_size_schedule]
# learning_rate = tf.train.piecewise_constant(
#     tf.cast(global_step, tf.int32),
#     boundaries,
#     values)
# total_loss = model.loss
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)
train_step = tf.train.AdamOptimizer(args.lr).minimize(model.loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver.restore(sess, model_file)
# cifar = AugmentedCIFAR10Data(raw_cifar, sess, model)

epoch_num = 10001
batch_size = 128
num_checkpoints = 10000
noise_name = "noise" + str(ind_noise)
training_time = 0.0
for i in range(epoch_num):

    inds = random.sample(range(n), batch_size)
    x_batch, y_batch = x_train[inds, :], y_train[inds]
    start = timer()
    noise = attack.get_perturb(x_batch, y_batch, sess)
    end = timer()
    training_time += end - start
    
    
    adv_dict = {model.x_input: x_batch, model.y_input: y_batch, model.noise[noise_name]: noise}
    sess.run(train_step, feed_dict = adv_dict)
    
    if i % 100 == 0 and i>1:
        print ("noise" + str(ind_noise) + " eps " + str(args.eps))
        inds_eval = random.sample(range(10000), 1000)
        x_eval, y_eval = x_test[inds_eval, :], y_test[inds_eval]
        eval_dict = {model.x_input: x_eval, model.y_input: y_eval}
        print('average examples per second: {:.4}'.format(batch_size*100./training_time))
        print ("iter: " + str(i) + ", epoch: "+ str(i * batch_size / 10000.) + ", accuracy: " + str(sess.run(model.accuracy, feed_dict= eval_dict)))
        training_time = 0.

    if i % num_checkpoints == 0:
        save_name = 'checkpoint_' + str(ind_noise) + '_eps_' + str(args.eps) + '_iter_' + str(i)
        saver.save(sess,
                   os.path.join(model_dir, save_name),
                   global_step=global_step)
sess.close()
    
    
    
    