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

import cifar10_input

class LinfPGDAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
             size a, while always staying within epsilon from the initial
             point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.beta1 = 0.9
        self.beta2 = 0.99

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

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
    
class LinfProjAdamAttack:
    def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func):
        """Attack parameter initialization. The attack performs k steps of
             size a, while always staying within epsilon from the initial
             point."""
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.rand = random_start
        self.beta1 = 0.9
        self.beta2 = 0.99

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]
#         self.ll_grad = tf.gradients(model.ll_loss, model.x_input)[0]

    def proj_adam_att(self, x_nat, y, sess, steps = None):
        """Given a set of examples (x_nat, y), returns a set of adversarial
             examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 255) # ensure valid pixel range
        else:
            x = np.copy(x_nat)

        batch_data = np.copy(x_nat)
        V = np.zeros(x.shape)
        V_hat = np.zeros(x.shape)
        m = np.zeros(x.shape)
        beta1 = 0.9
        beta2 = 0.99

        if not steps:
            steps = self.num_steps
        for i in range(steps):
            
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x, self.model.y_input: y})
            
            m = m * beta1 + grad * (1 - beta1)
            V = V * beta1 + grad**2 * (1 - beta2)
            V_hat = np.maximum(V, V_hat)
            x = np.add(x, self.step_size * np.sign(m/np.sqrt(V_hat+ 1e-8)), out=x, casting='unsafe')
            
#             x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

#             v = - 1* self.epsilon * np.sign(grad) + batch_data
#             d = v - x
#             g = np.sum(d* -grad)
#             print (g)  
#             x -= self.step_size * d


            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0, 255) # ensure valid pixel range

        return x
    
    



if __name__ == '__main__':
    import json
    import sys
    import math


    from model import Model

    with open('config.json') as config_file:
        config = json.load(config_file)

    model_file = tf.train.latest_checkpoint(config['model_dir'])
    if model_file is None:
        print('No model found')
        sys.exit()

    model = Model(mode='eval')
    attack = LinfPGDAttack(model,
                             config['epsilon'],
                             config['num_steps'],
                             config['step_size'],
                             config['random_start'],
                             config['loss_func'])
    saver = tf.train.Saver()

    data_path = config['data_path']
    cifar = cifar10_input.CIFAR10Data(data_path)

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

            x_batch = cifar.eval_data.xs[bstart:bend, :]
            y_batch = cifar.eval_data.ys[bstart:bend]

            x_batch_adv = attack.perturb(x_batch, y_batch, sess)

            x_adv.append(x_batch_adv)

            nat_dict_test = {model.x_input: x_batch,
                        model.y_input: y_batch}
            adv_dict_test = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}
            nat_full += sess.run(model.num_correct, feed_dict=nat_dict_test)
            adv_full += sess.run(model.num_correct, feed_dict=adv_dict_test)

            print ('batch ', ibatch, nat_full, adv_full)
            
        

        print('Storing examples')
        path = config['store_adv_path']
        x_adv = np.concatenate(x_adv, axis=0)
        np.save(path, x_adv)
        print('Examples stored in {}'.format(path))
        
        nat_full_acc = nat_full/num_eval_examples
        adv_full_acc = adv_full/num_eval_examples

        print('Finish acc :', nat_full_acc, adv_full_acc )
        print ()