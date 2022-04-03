# -*- coding: utf-8 -*-
"""
ExamGAN- train
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from examgan_model import ExamGAN
from examgan_data import GetDataset

class Trainer(ExamGAN):
    
    def __init__(self,mode,batch_size,ebank_size,skill_num,
                 h1_size,h2_size,max_epoch,keep_prob):
        ExamGAN.__init__(self,mode,batch_size,ebank_size,
                         skill_num,h1_size,h2_size,max_epoch,keep_prob)
        self.__keep_prob = keep_prob
    
    def getzsize(self):
        return ExamGAN.getes_size(self)
    
    def getessize(self):
        return ExamGAN.getes_size(self)
    
    def exec_train(self):
        z,x,y = ExamGAN.placeholder(self)
        x_generated,g_params = ExamGAN.generator(self,z,y)  # 生成器生成样本
        d_prob_real,d_prob_fake,d_params = ExamGAN.discriminator(self,x,x_generated,self.__keep_prob,y)  # 鉴别器对生成样本进行鉴别
        return z,x,y,x_generated,g_params,d_prob_real,d_prob_fake,d_params


def main():
    # paramenters
    data_path = 'dataset\\'
    DSname = 'assistments0910'
    mode = 'assistments0910'
    skill_num=124
    is_train = True
    rate = 0.97
    batch_size = 1940
    ebank_size = 10000
    h1_size = 128
    h2_size = 256
    max_epoch = 2000
    keep_prob = 0.5
    
    
    Tr = Trainer(mode,batch_size,ebank_size,skill_num,h1_size,h2_size,max_epoch,keep_prob)  
    z_size = Tr.getes_size()
    es_size = Tr.getes_size()
    z,x,y,x_generated,g_params,d_prob_real,d_prob_fake,d_params = Tr.exec_train()
    

    # loss
    d_loss = -tf.reduce_mean(tf.log(d_prob_real+1e-30) + tf.log(1.-d_prob_fake+1e-30))  
    g_loss = -tf.reduce_mean(tf.log(d_prob_fake+1e-30))
    
    
    # optimization
    g_solver = tf.train.GradientDescentOptimizer(0.001).minimize(g_loss,var_list=g_params) 
    d_solver = tf.train.GradientDescentOptimizer(0.001).minimize(d_loss,var_list=d_params) 

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    
    # get data
    data = GetDataset(data_path, DSname, skill_num, is_train, rate)
    x_train, y_train = data.gettrainData()
    x_test, y_test = data.gettestData()
    
    def get_z(shape):
        return np.random.uniform(0.,1.,size=shape).astype(np.float32)  
    
    # train
    for i in range(max_epoch):
    # sample
        if i>0 and i % 600 == 0:
            for j in range(60):
                cond_y = y_test[j].reshape(1,skill_num*2)
                sample = sess.run(x_generated, feed_dict = {z:get_z([1,z_size]),y:cond_y})
                sample = sample.reshape(-1,es_size)
                if j % 20 == 0 :
                    np.savetxt('samples\\' + mode +'\\examgan\\%d_%d.csv'%(i,j), sample, delimiter = ',',fmt='%r')

    # major training
    x_mb,y_mb = x_train, y_train
    
    _,d_loss_ = sess.run([d_solver,d_loss],feed_dict={x:x_mb,z:get_z([batch_size,z_size]),y:y_mb.astype(np.float32)})
    _,g_loss_ = sess.run([g_solver,g_loss],feed_dict={z:get_z([batch_size,z_size]),y:y_mb.astype(np.float32)})
    saver = tf.train.Saver(g_params)
    saver.save(sess, 'model\\' + mode, global_step=500, write_meta_graph = False) 

if __name__=='__main__':
    main()
        
        
        
        