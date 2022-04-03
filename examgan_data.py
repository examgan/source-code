# -*- coding: utf-8 -*-
"""
ExamGAN- Getting data
"""

import numpy as np
import pandas as pd
import os
from itertools import chain

class GetDataset:
    def __init__(self,data_path,DSname,skill_num,is_train,rate):
        self.__dirpath = data_path
        self.__dataset = DSname
        self.__skillnum = skill_num
        self.__rate = rate
    
    def getItem(self):
        path_trs = self.getStudent()
        skill_num = self.getskillnum()
        esdata = []
        cdata = []
        for st_filename in os.listdir(path_trs):
            sp = np.array(pd.read_csv(path_trs+st_filename,header=None))
            c = np.zeros((2,skill_num))
            for i in range(sp.shape[1]):
                c[0,i] = np.mean(sp[:,i])
                c[1,i] = np.var(sp[:,i])
            cc = list(chain.from_iterable(zip(c[0], c[1])))
            for i in range(20):
                cdata.append(cc)
            path_tre = self.getExamscript(st_filename)
            for es_filename in os.listdir(path_tre):
                eg = np.array(pd.read_csv(path_tre+es_filename,header=None))
                eg = list(eg[0])
                esdata.append(eg)
        return np.array(cdata), np.array(esdata)
    def getpath(self):
        return self.__dirpath
    def getdata(self):
        return self.__dataset
    def getskillnum(self):
        return self.__skillnum
    def getStudent(self): 
        return self.__dirpath + self.__dataset + '\\st\\'
    def getExamscript(self, st_filename):
        return self.__dirpath + self.__dataset + '\\es\\' + st_filename[:-4] +'\\'
    def gettrainData(self):
        rate = self.__rate
        cdata, esdata = self.getItem()
        y_train = cdata[:int(rate*cdata.shape[0]),:] 
        print ('train conditions is:')
        print (y_train.shape)
        x_train = esdata[:int(rate*esdata.shape[0]),:]         
        print ('train sample is:')
        print (x_train.shape)
        return x_train, y_train
    
    def gettestData(self):
        rate = self.__rate
        cdata, esdata = self.getItem()
        y_test = cdata[int(rate*cdata.shape[0]):,:] 
        print ('test conditions is:')
        print (y_test.shape)
        x_test = esdata[int(rate*esdata.shape[0]):,:]
        print ('test sample is:')
        print (x_test.shape)
        return x_test, y_test

def main():
    data_path = 'dataset\\'
    DSname = 'assistments0910'
    skill_num=124
    is_train = True
    rate = 0.97
    data = GetDataset(data_path, DSname, skill_num, is_train, rate)
    traindata = data.gettrainData()
    testdata = data.gettestData()
    return traindata, testdata

if __name__=='__main__':
    main()
    
    

