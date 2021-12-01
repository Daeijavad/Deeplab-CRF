import tensorflow as tf
import pydensecrf.densecrf as dcrf
from cv2 import imread, imwrite
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import numpy as np


class Network(object):
    
    dilations = [[1, 1], [1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [12], [1], [1]]
    num_class = 21
    
    def __init__(self, network, weights=None):
        self.network = network
        self.network_variables = []
        if weights is None:
            for i, j in network:
                if "/w" in i:
                    self.network_variables.append(tf.Variable(tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)(shape=j), name=i))
                else:
                    self.network_variables.append(tf.Variable(tf.constant_initializer(value=0.0, dtype=tf.float32)(shape=j), name=i))
        else:
            with open(weights, "rb") as file_:
                w = cPickle.load(f)
                for i, j in network:
                    self.network_variables.append(tf.Variable(w[i], name=i))
            
    def create_network(self, input_batch, keep_prob):
        current = input_batch
        ks=3
        v_idx = 0        
        for b_idx in xrange(len(Network.dilations) - 1):
            for l_idx, dilation in enumerate(Network.dilations[b_idx]):
                w = self.network_variables[v_idx * 2]
                b = self.network_variables[v_idx * 2 + 1]
                if dilation == 1:
                    conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
                else:
                    conv = tf.nn.atrous_conv2d(current, w, dilation, padding='SAME')
                current = tf.nn.relu(tf.nn.bias_add(conv, b))
                v_idx += 1
            if b_idx < 3:
                current = tf.nn.max_pool(current, ksize=[1, ks, ks, 1], strides=[1, 2, 2, 1], padding='SAME')
            elif b_idx == 3:
                current = tf.nn.max_pool(current, ksize=[1, ks, ks, 1], strides=[1, 1, 1, 1], padding='SAME')
            elif b_idx == 4:
                current = tf.nn.max_pool(current, ksize=[1, ks, ks, 1], strides=[1, 1, 1, 1], padding='SAME')
                current = tf.nn.avg_pool(current, ksize=[1, ks, ks, 1], strides=[1, 1, 1, 1], padding='SAME')
            elif b_idx <= 6:
                current = tf.nn.dropout(current, keep_prob=keep_prob)
        
        w = self.network_variables[v_idx * 2]
        b = self.network_variables[v_idx * 2 + 1]
        conv = tf.nn.conv2d(current, w, strides=[1, 1, 1, 1], padding='SAME')
        current = tf.nn.bias_add(conv, b)

        return current
    
    def prepare_label(self, input_batch, new_size):
        with tf.name_scope('label_encode'):
            input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)
            input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) 
            input_batch = tf.one_hot(input_batch, depth=21)
        return input_batch
    
    def preds(self, input_batch):
        raw_output = self.create_network(tf.cast(input_batch, tf.float32), keep_prob=tf.constant(1.0))
        raw_output = tf.image.resize_bilinear(raw_output, tf.shape(input_batch)[1:3,])
        raw_output = tf.argmax(raw_output, dimension=3)
        raw_output = tf.expand_dims(raw_output, dim=3)
        return tf.cast(raw_output, tf.uint8)
        
    
    def loss(self, img_batch, label_batch):
        raw_output = self.create_network(tf.cast(img_batch, tf.float32), keep_prob=tf.constant(0.5))
        prediction = tf.reshape(raw_output, [-1, Network.num_class])
        
        label_batch = self.prepare_label(label_batch, tf.pack(raw_output.get_shape()[1:3]))
        gt = tf.reshape(label_batch, [-1, Network.num_class])
        
        loss = tf.nn.softmax_cross_entropy_with_logits(prediction, gt)
        reduced_loss = tf.reduce_mean(loss)
        
        return reduced_loss

def CRFlayer(imginput, maskinput):
    mask_rgb = maskinput.astype(np.uint32)
    mask_lbl = mask_rgb[:,:,0] + (mask_rgb[:,:,1] << 8) + (mask_rgb[:,:,2] << 16)

    colors, labels = np.unique(mask_lbl, return_inverse=True)
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    n_labels = len(set(labels.flat))

    d = dcrf.DenseCRF2D(imginput.shape[1], imginput.shape[0], n_labels)
    U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=0)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=(3, 3), compat=3)
    d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=imginput, compat=10)

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)
    MAP = colorize[MAP,:]
    maskoutput = MAP.reshape(imginput.shape)

    return maskoutput
