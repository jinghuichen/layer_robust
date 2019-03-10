from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, BatchNormalization
from keras import regularizers



class Model(object):
    def __init__(self):
        self.x_input = tf.placeholder(tf.float32, shape=[None, 3072])
        self.y_input = tf.placeholder(tf.int64, shape = [None])

        self.x_image = tf.reshape(self.x_input, [-1, 32, 32, 3])

#         # first convolutional layer
#         W_conv1 = self._weight_variable([3,3,3,64])
#         b_conv1 = self._bias_variable([64])

#         h_conv1 = tf.nn.relu(self._conv2d(self.x_image, W_conv1) + b_conv1)

#         # second convolutional layer
#         W_conv2 = self._weight_variable([3,3,64,64])
#         b_conv2 = self._bias_variable([64])

#         h_conv2 = tf.nn.relu(self._conv2d(h_conv1, W_conv2) + b_conv2)
#         h_pool2 = self._max_pool_2x2(h_conv2)

#         # third convolutional layer
#         W_conv3 = self._weight_variable([3, 3, 64, 128])
#         b_conv3 = self._bias_variable([128])

#         h_conv3 = tf.nn.relu(self._conv2d(h_pool2, W_conv3) + b_conv3)

#         # fourth convolutional layer
#         W_conv4 = self._weight_variable([3, 3, 128, 128])
#         b_conv4 = self._bias_variable([128])

#         h_conv4 = tf.nn.relu(self._conv2d(h_conv3, W_conv4) + b_conv4)
#         h_pool4 = self._max_pool_2x2(h_conv4)


#         # first fully connected layer
#         W_fc1 = self._weight_variable([8 * 8 * 128, 256])
#         b_fc1 = self._bias_variable([256])

#         h_pool4_flat = tf.reshape(h_pool4, [-1, 8 * 8 * 128])
#         h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

#         # second fully connected layer
#         W_fc2 = self._weight_variable([256, 256])
#         b_fc2 = self._bias_variable([256])

#         h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

#         # output layer
#         W_fc3 = self._weight_variable([256,10])
#         b_fc3 = self._bias_variable([10])

#         self.pre_softmax = tf.matmul(h_fc2, W_fc3) + b_fc3

        
        self.conv1 = Conv2D(64, (3, 3), padding='same', name='block1_conv1', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.x_image)
        self.noise1 = tf.placeholder_with_default(tf.zeros_like(self.conv1), shape=self.conv1.shape)
        self.conv1n = self.conv1 + self.noise1
        self.bn1 = BatchNormalization()(self.conv1n)
        self.relu1 = Activation('relu')(self.bn1)
        
        self.conv2 = Conv2D(64, (3, 3), padding='same', name='block1_conv2', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.relu1)
        self.bn2 = BatchNormalization()(self.conv2)
        self.relu2 = Activation('relu')(self.bn2)
        self.pooling2 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool1')(self.relu2)

        self.conv3 = Conv2D(128, (3, 3), padding='same', name='block2_conv1', 
                   kernel_regularizer=regularizers.l2(0.0002))(self.pooling2)
        self.bn3 = BatchNormalization()(self.conv3)
        self.relu3 = Activation('relu')(self.bn3)
        
        self.conv4 = Conv2D(128, (3, 3), padding='same', name='block2_conv2', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.relu3)
        self.bn4 = BatchNormalization()(self.conv4)
        self.relu4 = Activation('relu')(self.bn4)
        self.pooling4 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool1')(self.relu4)

        self.conv5 = Conv2D(196, (3, 3), padding='same', name='block3_conv1', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.pooling4)
        self.bn5 = BatchNormalization()(self.conv5)
        self.relu5 = Activation('relu')(self.bn5)
        
        self.conv6 = Conv2D(196, (3, 3), padding='same', name='block3_conv2', 
                            kernel_regularizer=regularizers.l2(0.0002))(self.relu5)
        self.bn6 = BatchNormalization()(self.conv6)
        self.relu6 = Activation('relu')(self.bn6)
        self.pooling6 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool1')(self.relu6)

        self.flatten = Flatten(name='flatten')(self.pooling6)

        self.dense1 = Dense(256, kernel_regularizer=regularizers.l2(0.0002))(self.flatten)
        self.bn_dense = BatchNormalization()(self.dense1)
        self.relu_dense = Activation('relu')(self.bn_dense)

        self.dense2 = Dense(10, name='logits', kernel_regularizer=regularizers.l2(0.0002))(self.relu_dense)


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
    
    