# -*- coding: utf-8 -*-
"""
ExamGAN- show
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Get_examscript:
    def __init__(self,mode,skill_num,id_es,es_size):
        self.__mode = mode
        self.__skill_num = skill_num
        self.__id_es = id_es
        self.__es_size = es_size
    
    def get_es(self):
        qdb = np.array(pd.read_csv('qbank\\qb_assistments0910.csv'))
        qdbs = np.array(pd.read_csv('samples\\assistments0910\\examgan\\%s.csv'%self.__id_es,header=None))[0] 
        ind = np.array(np.argpartition(qdbs,self.__es_size)[-self.__es_size:])
        es = qdb[ind,1:self.__skill_num+1]
        return es

def main():
    mode = 'assistments0910'
    skill_num = 124
    es_size = 100
    id_es = 600
    for i in range(3):
        for j in range(3):
            id_es = str(i*600)+'_'+str(j*20)  
            Ges = Get_examscript(mode,skill_num,id_es,es_size)
            es = Ges.get_es()
            plt.imshow(es, cmap = 'winter')
            plt.savefig('img\\es_%s.png'%id_es)
    
if __name__=='__main__':
    main()
    