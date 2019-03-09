"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

from model import Model
import cifar10_input
from pgd_attack import LinfPGDAttack, LinfProjAdamAttack
import math
import pickle
from utils import progress_bar

with open('config.json') as config_file:
    config = json.load(config_file)

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

# Setting up training parameters
max_num_epochs = config['max_num_epochs']
num_output_steps = config['num_output_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']
test_batch_size = batch_size 

nat_accs = []
adv_accs = []

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent
# + weight_decay * model.weight_decay_loss
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)

# Set up adversary
attack_train = LinfProjAdamAttack(model,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])
attack_eval = LinfPGDAttack(model,
                       config['epsilon'],
                       config['eval_num_steps'],
                       config['eval_step_size'],
                       config['random_start'],
                       config['loss_func'])

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


saver = tf.train.Saver(max_to_keep=3)

# keep the configuration file with the model for reproducibility
shutil.copy('config.json', model_dir)

with tf.Session() as sess:

    # initialize data augmentation
    cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)
 
    sess.run(tf.global_variables_initializer())
    training_time = 0.0

    # Main training loop
    for epoch in range(max_num_epochs):
        nat_acc = []
        adv_acc = []
        xent = []
        total_batches = math.ceil(raw_cifar.train_data.n/batch_size)
        for b in range(total_batches):
            x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                               multiple_passes=True)

            
# #             One step attack
#             start = timer()
#             x_batch_adv = attack_train.proj_adam_att(x_batch, y_batch, sess, 1)
#             end = timer()
#             training_time += end - start
#             nat_dict = {model.x_input: x_batch,
#                         model.y_input: y_batch}
#             top2 = sess.run(model.top2, feed_dict=nat_dict)
#             conf = [t1 - t2 for t1,t2 in top2]
#             avg_conf = np.mean(conf)
#             print (avg_conf, int(config['num_steps'] * avg_conf))
            
            # Compute Additonal Adversarial Perturbations
            start = timer()
#             x_batch_adv = attack_train.proj_adam_att(x_batch, y_batch, sess, int(config['num_steps'] * avg_conf))
            x_batch_adv = attack_train.proj_adam_att(x_batch, y_batch, sess)
            end = timer()
            training_time += end - start
            nat_dict = {model.x_input: x_batch,
                        model.y_input: y_batch}
            adv_dict = {model.x_input: x_batch_adv,
                        model.y_input: y_batch}

            
            
            xent.append(sess.run(model.xent, feed_dict=nat_dict))
            nat_acc.append(sess.run(model.accuracy, feed_dict=nat_dict))
            adv_acc.append(sess.run(model.accuracy, feed_dict=adv_dict))
            progress_bar(b, total_batches, 'nat_acc: %.3f%%  | adv_acc: %.3f%%  | loss: %.3f  '
                        % (np.mean(nat_acc) * 100, np.mean(adv_acc) * 100, np.mean(xent)))
 
    
        #     # Output to stdout
        #     if ii % num_output_steps == 0:
        #         nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
        #         adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
        #         print('Step {}:    ({})'.format(ii, datetime.now()))
        #         print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
        #         print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
        #         if ii != 0:
        #         print('    {} examples per second'.format(
        #             num_output_steps * batch_size / training_time))
        #         training_time = 0.0
 
            # Write a checkpoint
            if b % num_checkpoint_steps == 0:
                saver.save(sess,
                         os.path.join(model_dir, 'checkpoint'),
                         global_step=global_step)

            # Actual training step
            start = timer()
            sess.run(train_step, feed_dict=adv_dict)
            end = timer()
            training_time += end - start

            
            
            
        #Evaluate after one epoch
        nat_full = 0
        adv_full = 0
        test_batches = math.ceil(raw_cifar.eval_data.n/test_batch_size)
        for b in range(test_batches):
            x_batch_test, y_batch_test = cifar.eval_data.get_next_batch(test_batch_size,
                                                               multiple_passes=True)
            x_batch_adv_test = attack_eval.perturb(x_batch_test, y_batch_test, sess)
            nat_dict_test = {model.x_input: x_batch_test,
                        model.y_input: y_batch_test}
            adv_dict_test = {model.x_input: x_batch_adv_test,
                        model.y_input: y_batch_test}
            nat_full += sess.run(model.num_correct, feed_dict=nat_dict_test)
            adv_full += sess.run(model.num_correct, feed_dict=adv_dict_test)
            
            progress_bar(b, test_batches, 'nat_acc: %.3f%%  | adv_acc: %.3f%% '
                % (nat_full/ raw_cifar.eval_data.n * 100, adv_full/ raw_cifar.eval_data.n * 100 ))

        nat_full_acc = nat_full/raw_cifar.eval_data.n
        adv_full_acc = adv_full/raw_cifar.eval_data.n
        nat_accs.append(nat_full_acc)
        adv_accs.append(adv_full_acc)
        print('Finish Epoch {} : Total test nat acc {:.4}%  adv accuracy {:.4}% ({})'.format(epoch,nat_full_acc * 100, adv_full_acc * 100, datetime.now()))
        print ()
        with open('training_natacc' +  '.txt', 'w') as f: 
            json.dump(nat_accs, f)
        with open('training_advacc' +  '.txt', 'w') as f: 
            json.dump(adv_accs, f)
  


 