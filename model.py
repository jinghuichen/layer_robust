from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, BatchNormalization
from keras import regularizers



class Model(object):
    def __init__(self):
        self.noise = {}
        self.x_input = tf.placeholder(tf.float32, shape=[None, 3072])
        self.noise["noise0"] = tf.placeholder_with_default(tf.zeros_like(self.x_input), shape=self.x_input.shape)
        self.x_inputn = self.x_input + self.noise["noise0"]
        
        
        self.y_input = tf.placeholder(tf.int64, shape = [None])

        self.x_image = tf.reshape(self.x_inputn, [-1, 32, 32, 3])
        
        
        self.conv1 = Conv2D(64, (3, 3), padding='same', name='block1_conv1', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.x_image)
        
       # self.conv1n = self.conv1 
        self.bn1 = BatchNormalization()(self.conv1)
        self.relu1 = Activation('relu')(self.bn1)
        self.noise["noise1"] = tf.placeholder_with_default(tf.zeros_like(self.relu1), shape=self.relu1.shape)
        self.relun1 = self.relu1 + self.noise["noise1"]
        
        self.conv2 = Conv2D(64, (3, 3), padding='same', name='block1_conv2', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.relun1)
        self.bn2 = BatchNormalization()(self.conv2)
        self.relu2 = Activation('relu')(self.bn2)
        self.pooling2 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(self.relu2)
        self.noise["noise2"] = tf.placeholder_with_default(tf.zeros_like(self.pooling2), shape=self.pooling2.shape)
        self.poolingn2 = self.pooling2 + self.noise["noise2"]

        self.conv3 = Conv2D(128, (3, 3), padding='same', name='block2_conv1', 
                   kernel_regularizer=regularizers.l2(0.0002))(self.poolingn2)
        self.bn3 = BatchNormalization()(self.conv3)
        self.relu3 = Activation('relu')(self.bn3)
        self.noise["noise3"] = tf.placeholder_with_default(tf.zeros_like(self.relu3), shape=self.relu3.shape)
        self.relun3 = self.relu3 + self.noise["noise3"]
        
        self.conv4 = Conv2D(128, (3, 3), padding='same', name='block2_conv2', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.relun3)
        self.bn4 = BatchNormalization()(self.conv4)
        self.relu4 = Activation('relu')(self.bn4)
        self.pooling4 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool1')(self.relu4)
        self.noise["noise4"] = tf.placeholder_with_default(tf.zeros_like(self.pooling4), shape=self.pooling4.shape)
        self.poolingn4 = self.pooling4 + self.noise["noise4"]

        self.conv5 = Conv2D(196, (3, 3), padding='same', name='block3_conv1', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.poolingn4)
        self.bn5 = BatchNormalization()(self.conv5)
        self.relu5 = Activation('relu')(self.bn5)
        self.noise["noise5"] = tf.placeholder_with_default(tf.zeros_like(self.relu5), shape=self.relu5.shape)
        self.relun5 = self.relu5 + self.noise["noise5"]
        
        
        self.conv6 = Conv2D(196, (3, 3), padding='same', name='block3_conv2', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.relun5)
        self.bn6 = BatchNormalization()(self.conv6)
        self.relu6 = Activation('relu')(self.bn6)
        self.pooling6 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool1')(self.relu6)
        self.noise["noise6"] = tf.placeholder_with_default(tf.zeros_like(self.pooling6), shape=self.pooling6.shape)
        self.poolingn6 = self.pooling6 + self.noise["noise6"]        

        self.flatten = Flatten(name='flatten')(self.poolingn6)

        self.dense1 = Dense(256, kernel_regularizer=regularizers.l2(0.0002))(self.flatten)
        self.bn_dense = BatchNormalization()(self.dense1)
        self.relu_dense = Activation('relu')(self.bn_dense)
        self.noise["noise7"] = tf.placeholder_with_default(tf.zeros_like(self.relu_dense), shape=self.relu_dense.shape)
        self.relu_densen = self.relu_dense + self.noise["noise7"]
        
        self.dense2 = Dense(10, name='logits', kernel_regularizer=regularizers.l2(0.0002))(self.relu_densen)


        self.pre_softmax = self.dense2

        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y_input, logits=self.pre_softmax)

        self.loss = tf.reduce_sum(self.y_xent)

        self.y_pred = tf.argmax(self.pre_softmax, 1)

        self.correct_prediction = tf.equal(self.y_pred, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def _max_pool_2x2( x):
        return tf.nn.max_pool(x,
                              ksize = [1,2,2,1],
                              strides=[1,2,2,1],
                              padding='SAME')
    
    
