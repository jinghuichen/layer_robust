
# coding: utf-8

# In[1]:


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
            
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x, self.model.y_input: y})
            x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255) # ensure valid pixel range

        return x
    
def linf_norm(x):
    d = 1
    dd = x.shape
    for i in range(len(dd)-1):
        d *= dd[i+1]
    x = x.reshape([dd[0], d])
    return np.mean(np.max(np.abs(x), axis = 1))

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
        print(model_file)
        saver.restore(sess, model_file)

        # Iterate over the samples batch-by-batch
        num_eval_examples = config['num_eval_examples']
        eval_batch_size = config['eval_batch_size']
        num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

        x_adv = [] # adv accumulator
        linf_adv1, linf_adv2, linf_adv3, linf_adv4, linf_adv5, linf_adv6, linf_adv7  = [],[],[],[],[],[],[] #adv distance accumulator        
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
            
            nat_h1, nat_h2, nat_h3, nat_h4, nat_h5, nat_h6, nat_h7 = sess.run(
                [model.relu1, model.pooling2, model.relu3, model.pooling4, model.relu5, model.pooling6, model.relu_dense],
                    feed_dict = nat_dict_test)
            adv_h1, adv_h2, adv_h3, adv_h4, adv_h5, adv_h6, adv_h7 = sess.run(
                [model.relu1, model.pooling2, model.relu3, model.pooling4, model.relu5, model.pooling6, model.relu_dense],
                    feed_dict = adv_dict_test)
            
            linf_adv1.append(linf_norm(adv_h1 - nat_h1))
            linf_adv2.append(linf_norm(adv_h2 - nat_h2))
            linf_adv3.append(linf_norm(adv_h3 - nat_h3))
            linf_adv4.append(linf_norm(adv_h4 - nat_h4))
            linf_adv5.append(linf_norm(adv_h5 - nat_h5))
            linf_adv6.append(linf_norm(adv_h6 - nat_h6))
            linf_adv7.append(linf_norm(adv_h7 - nat_h7))
            
            
#             print (nat_h1.shape, nat_h2.shape, nat_h3.shape, nat_h4.shape, nat_h5.shape, nat_h6.shape, nat_h7.shape)
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
        eps_budget = [np.mean(linf_adv1), np.mean(linf_adv2), np.mean(linf_adv3), np.mean(linf_adv4),
                     np.mean(linf_adv5), np.mean(linf_adv6), np.mean(linf_adv7)]
        print ('epsilon for each layer :', eps_budget)
        path = 'eps_layer.npy'
        np.save(path, eps_budget)
        print('Eps budget stored in {}'.format(path))

        print('Finish acc :', nat_full_acc, adv_full_acc )
        print ()

