"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow as tf
import numpy as np
from cifar_setup import *
import random

class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start):
        """Attack parameter initialization. The attack performs k steps of
             size a, while always staying within epsilon from the initial
             point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        
        loss = model.loss

        self.grad = tf.gradients(loss, model.x_input)[0]
        self.grad1= tf.gradients(loss, model.noise1)[0]
 
 
    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
             examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 255) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        batch_data = np.copy(x_nat)
        
        for i in range(self.num_steps):
            
#             grad = sess.run(self.grad, feed_dict={self.model.x_input: x, self.model.y_input: y})
#             x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')
#             x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
#             x = np.clip(x, 0, 255) # ensure valid pixel range

            if i==0:
                conv1 = sess.run(self.model.conv1, feed_dict={self.model.x_input: x, self.model.y_input: y})
                shape = conv1.shape
                noise = np.zeros(shape)
                noise_nat = noise
            grad = sess.run(self.grad1, feed_dict={self.model.x_input: x, self.model.y_input: y, self.model.noise1:noise})
            np.add(noise, self.step_size * np.sign(grad), out=noise, casting='unsafe')
            noise = np.clip(noise, noise_nat - self.epsilon, noise_nat + self.epsilon)
            print (noise)

        return x
    


if __name__ == '__main__':
    import json
    import sys
    import math


    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint('./model')
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model()
    attack = LinfPGDAttack(model,
                           epsilon = 0.031/5,
                           num_steps = 20,
                           step_size = 0.031/10,
                           random_start = True)
    saver = tf.train.Saver()


    data_path = './data_set/cifar_10/'
    (x_train, y_train), (x_test, y_test) = get_data_set("train"), get_data_set("test")
    n = y_train.shape[0]
    with tf.Session() as sess:
        # Restore the checkpoint
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_adv = [] # adv accumulator
                
        #Evaluate accuracy
        nat_full = 0
        adv_full = 0

        print('Iterating over {} batches'.format(num_batches))

        for ibatch in range(num_batches):
            
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)
            print('batch size: {}'.format(bend - bstart))
     
            x_batch, y_batch = x_test[bstart:bend, :], y_test[bstart:bend]
#             print (y_batch)

            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            x_adv.append(x_batch_adv)

            nat_dict_test = {model.x_input: x_batch,
                             model.y_input: y_batch}
            adv_dict_test = {model.x_input: x_batch_adv,
                             model.y_input: y_batch}
            nat_full += sess.run(model.num_correct, feed_dict=nat_dict_test)
#             print (sess.run(model.y_pred, feed_dict=nat_dict_test))
#             print (sess.run(model.accuracy, feed_dict=nat_dict_test))
#             print (sess.run(model.num_correct, feed_dict=nat_dict_test))
            adv_full += sess.run(model.num_correct, feed_dict=adv_dict_test)

            print ('batch: ', ibatch, ' nat acc: ', nat_full, '/', bend, '(', nat_full/bend, ')' , ' adv acc: ', adv_full, '/', bend, '(', adv_full/bend, ')' )
            
        

        print('Storing examples')
        path = config['store_adv_path']
        x_adv = np.concatenate(x_adv, axis=0)
        np.save(path, x_adv)
        print('Examples stored in {}'.format(path))
        
        nat_full_acc = nat_full/num_eval_examples
        adv_full_acc = adv_full/num_eval_examples

        print('Finish acc :', nat_full_acc, adv_full_acc )
        print ()