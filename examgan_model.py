# -*- coding: utf-8 -*-
"""
ExamGAN Model
"""

import tensorflow as tf
import numpy as np

class ExamGAN():
    """Parameters of exam script
    """
    def __init__(self,mode,batch_size,ebank_size,skill_num,h1_size,h2_size,max_epoch,keep_prob):
        self.__mode = mode
        self.__es_size = ebank_size 
        self.__cond_size = skill_num * 2 
        self.__batch_size = batch_size 
        self.__h1_size = h1_size 
        self.__h2_size = h2_size 
        self.__max_epoch = max_epoch 
        self.__keep_prob = keep_prob  
    def es_cond_connect(self):
        es_size = self.getes_size()
        cond_size = self.getcond_size()
        return es_size + cond_size
    def z_cond_connect(self):
        z_size = self.getes_size()
        cond_size = self.getcond_size()
        return z_size + cond_size
    def getmode(self):
        return self.__mode
    def getes_size(self):
        return self.__es_size
    def getcond_size(self):
        return self.__cond_size
    def getbatch_size(self):
        return self.__batch_size
    def geth1_size(self):
        return self.__h1_size
    def geth2_size(self):
        return self.__h2_size
    def max_epoch(self):
        return self.__max_epoch
    def keep_prob(self):
        return self.__keep_prob
    
    def placeholder(self):
        z_size = self.__es_size
        es_size = self.__es_size
        cond_size = self.__cond_size
        z = tf.placeholder(tf.float32,shape=[None,z_size]) 
        x = tf.placeholder(tf.float32,shape=[None,es_size]) 
        y = tf.placeholder(tf.float32,shape=[None,cond_size])
        return z,x,y

    def xavier_init(self,shape):
        in_dim = shape[0]
        stddev = 1./tf.sqrt(in_dim/2.)
        return tf.random_normal(shape=shape,stddev=stddev)

    def generator(self,z,y):
        z_cond = tf.concat([z,y],axis=1)
        z_cond_size = self.z_cond_connect()
        h1_size = self.__h1_size
        es_size = self.__es_size
        # L1
        w1 = tf.Variable(self.xavier_init([z_cond_size,h1_size])) 
        b1 = tf.Variable(tf.zeros([h1_size]),dtype=tf.float32) 
        h1 = tf.nn.sigmoid(tf.matmul(z_cond,w1)+b1) 
        # Out
        w2 = tf.Variable(self.xavier_init([h1_size,es_size])) 
        b2 = tf.Variable(tf.zeros([es_size]),dtype=tf.float32) 
        x_generated = tf.nn.tanh(tf.matmul(h1,w2)+b2) 
        params = [w1,b1,w2,b2]
        return x_generated, params  
    

    def discriminator(self,x,x_generated,keep_prob,y):
        x_cond = tf.concat([x,y],axis=1)  
        x_generated_cond = tf.concat([x_generated,y],axis=1)  
        keep_prob = self.__keep_prob
        es_cond_size = self.es_cond_connect()
        h1_size = self.__h1_size
        # L1
        w1 = tf.Variable(self.xavier_init([es_cond_size,h1_size])) 
        b1 = tf.Variable(tf.zeros([h1_size]),dtype=tf.float32) 
        h1_x = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(x_cond,w1)+b1),keep_prob) 
        h1_x_generated = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(x_generated_cond,w1)+b1),keep_prob) 
        # Out
        w2 = tf.Variable(self.xavier_init([h1_size,1]))
        b2 = tf.Variable(tf.zeros([1]),dtype=tf.float32)
        # last layer
        d_prob_x = tf.nn.sigmoid(tf.matmul(h1_x,w2)+b2) 
        d_prob_x_generated = tf.nn.sigmoid(tf.matmul(h1_x_generated,w2)+b2)
        params = [w1,b1,w2,b2]
        return d_prob_x,d_prob_x_generated,params

def main():
    mode = 'assistments0910'
    batch_size = 1940
    ebank_size = 10000
    skill_num = 124
    h1_size = 128
    h2_size = 256
    max_epoch = 2000
    keep_prob = 0.5
    
    ExamGAN(mode,batch_size,ebank_size,skill_num,h1_size,h2_size,max_epoch,keep_prob)
    
if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    