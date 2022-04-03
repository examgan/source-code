# -*- coding: utf-8 -*-
"""
ExamGAN- checking
"""

import pandas as pd
import numpy as np
import scipy.stats
import os
from functools import reduce

class Check:
    def __init__(self,skill_num,test_size,index_size,epoch_n):
        self.__skill_num = skill_num
        self.__test_size = test_size
        self.__index_size = index_size
        self.__epoch_n = epoch_n
    
    def getctest(self):
        c_path = 'dataset\\assistments0910\\st\\' 
        c_test = []
        for filename in os.listdir(c_path):
            ct = np.array(pd.read_csv(c_path+filename,header=None))
            c_test.append(ct)
        return c_test[-self.__test_size:]
    
    def getqbank(self):
        qdb = np.array(pd.read_csv('qbank\\qb_assistments0910.csv'))
        qdb_sum = qdb[:,1:self.__skill_num+1].sum(axis=0)
        qdb_pd = qdb_sum/np.sum(qdb_sum)
        return qdb, qdb_pd
        
    def checking(self, method):
        c_test = self.getctest()
        qdb,qdb_pd = self.getqbank()
        n = self.__epoch_n
        t = np.zeros([self.__test_size,self.__index_size])
        for i in range(t.shape[0]):
            t[i,0] = int(i)
            # Input students' skill mastery level
            c_test_i = c_test[i]
            g_es = np.array(pd.read_csv('samples\\assistments0910\\'+method+'\\%d_%d.csv'%(n,i*20),header=None))[0] # check positive samples
            ind = np.array(np.argpartition(g_es,-100)[-100:])
            es = qdb[ind,1:self.__skill_num+1]
            es_sum = es.sum(axis=0)
            es_pd = es_sum/np.sum(es_sum)
            Z = np.random.normal(70, 15, 50)
            R = []
            for s in range(c_test_i.shape[0]):
                r = 0
                for j in range(es.shape[0]):
                    r1 = np.multiply(c_test_i[s],es[j])
                    r2 = (r1[r1>0.1])
                    if r2.size == 0:
                        r4 = 0
                    else:
                        r3 = reduce(lambda a,b:a * b,r2)
                        r4 = 1 if r3>0 else 0
                    r = r + r4
                R = np.append(R,r)    
            # Difficulty
            t[i,1] = np.mean(R)/100
            # Validity-DISeucl Distinguishability
            t[i,2] = 1-np.linalg.norm(es_pd-qdb_pd)
            # Rationality
            t[i,3] = 1-scipy.stats.entropy(R,Z)
            # Distinguishability
            R = sorted(R)
            t[i,4] = (np.mean(R[-8:])-np.mean(R[:8]))/100
            print ('Dif:%f;Dista:%f;Div:%f;Disti:%f of exam script:%d'%(t[i,1],t[i,2],t[i,3],t[i,4],i))
        return t
    
def main():
    skill_num = 124
    test_size = 3
    index_size = 5
    epoch_n = 1200
    method = 'examgan'
    ck = Check(skill_num,test_size,index_size,epoch_n)
    ck.checking(method)

if __name__=='__main__':
    main()
    
        