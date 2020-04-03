#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os
os.system('pip install torch')
os.system('pip install torchaudio')
os.system('pip install lightgbm')
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

from speech_util.data_manager import DataManager
from speech_util.model_manager import ModelManager
from speech_util.tools import log, timeit
import numpy as np
config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = False
# (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
from speech_util.models import LogisticRegression
from speech_util.data_manager import lr_preprocess1 as lr_pre
from sklearn.preprocessing import StandardScaler
import random
from sklearn.tree import DecisionTreeClassifier
from speech_util.deepwis import *
import lightgbm as lgb

from sklearn.metrics import log_loss, roc_auc_score
def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    
LATE_STAGE_ROUND=-1
class Converter:
    def __init__(self,class_num,train_num,dataset):
        self.class_num=class_num
        self.train_num=train_num
        self.cnt=0
        self.dataset=dataset
        self.data_iter=self.dataset.make_one_shot_iterator().get_next()
        
        self.lr_data=None
        self.fetched_data_x=[]
        self.fetched_data_y=[]  
        
        self.class_index={}
        for i in range(class_num):
            self.class_index[i]=[]    
            
        self.is_converted=False
        self.is_multitag=False
#     @timeit
    def fetch(self,ratio=0.3):
        sample_num=max(int(ratio*self.train_num),100)
        if self.is_done():
            return 
        
        class_left=set()
        for i in range(self.class_num):
            if len(self.class_index[i])==0:
                class_left.add(i)
        print('before fetch :{} -> {}'.format(self.cnt,self.train_num))
        ct=0
        values,labels=[],[]
        while (ct < sample_num or len(class_left)!=0 ) and self.cnt<self.train_num:
            
            value,label=sess.run(self.data_iter)
#             ###for multilabel test
#             label[np.random.choice(range(len(label)),2)]=1
#             ###
            # for label control
            if not self.is_multitag:
                if np.sum(label>=1)>1:
                    self.is_multitag=True
            label_classes=np.where(label>=1)[0]
            for label_class in label_classes:
                self.class_index[label_class].append(self.cnt)
                class_left.discard(label_class)
            
            # for data store
            values.append(value.reshape((-1)))
            labels.append(label)
            
            self.cnt+=1
            ct+=1
            
        print('after fetch :{} -> {}'.format(self.cnt,self.train_num))    

        # for data store
        self.fetched_data_x.extend(values)
        self.fetched_data_y.extend(labels)
        
        # for data preprocess
        if self.lr_data is None:
            self.lr_data=lr_pre(np.array(values))
        else:
            self.lr_data=np.concatenate([self.lr_data,lr_pre(np.array(values))],axis=0)
       
        print("is multitag:{}".format(self.is_multitag))
          
#@timeit    
    def _get_reweighted_lr_data(self,sample_num):
        data=self.lr_data
        index=self.class_index
        class_num=self.class_num
        each_class_num=1+max(0,sample_num//class_num)
        selected_index=[]
        for i in range(class_num):
            if len(index[i])<each_class_num:
                multiples=each_class_num//len(index[i])
                left=each_class_num-multiples* len(index[i])
                selected_index.extend(index[i]*(multiples))
                selected_index.extend(random.sample(index[i],left))
            else:
                selected_index.extend(random.sample(index[i],each_class_num))
        random.shuffle(selected_index)
        print("reweighting : {} -> {}".format(len(self.lr_data),len(selected_index)))
        return self.lr_data[selected_index,:],np.asarray(self.fetched_data_y)[selected_index,:]
    def get_lr_data(self):
        sample_num=len(self.lr_data)
        data_x,data_y=self._get_reweighted_lr_data(sample_num)
        scaler = StandardScaler()
        X = scaler.fit_transform(data_x[:, :])
        return X,data_y
    def get_lr_pre_data(self):
        scaler = StandardScaler()
        X = scaler.fit_transform(self.lr_data[:, :])
        return X
    def get_dataset(self):
        if not self.is_converted:
            self.fetched_data_x,self.fetched_data_y=np.array(self.fetched_data_x),np.asarray(self.fetched_data_y)
            self.is_converted=True
        return self.fetched_data_x,self.fetched_data_y
    def is_done(self):
        return self.cnt>=self.train_num

def convert_speech(dataset, sess=tf.Session(config=config)):
    next_ele = dataset.make_one_shot_iterator().get_next()
    values = []
    labels = []
    while 1:
        try:
            value,label=sess.run(next_ele)
            values.append(value.reshape((-1)))
            labels.append(label)
        except tf.errors.OutOfRangeError as e:
            break
    labels=np.asarray(labels)
    print("--------------------------------------------------------------------")
    print("convert1 done")
    print("--------------------------------------------------------------------")
    return values, labels
def convert_metadata(metadata):
    tmp={}
    tmp['class_num']=metadata.get_output_size()
    tmp['train_num']=metadata.size()
    tmp['test_num']=metadata.size()
    return tmp

class speechModel(object):
    def __init__(self, metadata, train_output_path="./", test_input_path="./"):
        """ Initialization for model
        :param metadata: a dict formed like:
            {"class_num": 7,
             "train_num": 428,
             "test_num": 107,
             "time_budget": 1800}
        """
        self.done_training = False
        self.metadata = convert_metadata(metadata)
        self.train_loop_num = 0
        log('Metadata: ' + str(self.metadata))

        self.data_manager = None
        self.model_manager = None

        self.train_output_path = train_output_path
        self.test_input_path = test_input_path

        self.observed_train_dataset = None
        self.observed_test_dataset = None
        
        self.use_lr=True
        self.converter=None
        
        self.lr_cnt=0
        self.fetch_ratio=[0.01,0.4,0.7,0.5]
    
        set_random_seed_all(0xC0FFEE)
        self.model2nd=Model2(metadata,train_output_path,test_input_path)
        self.use2nd=False
        self.empty_times=14
        self.use2nd_start=0
        self.use2nd_test_ths=None
        
        self.lr=LogisticRegression()
        kwargs = {
                'kernel': 'liblinear',
                'max_iter': 100
            }
        self.lr.init_model(**kwargs)
        self.bst=None
        self.acc_lr=0
        self.acc_2nd=-1
        
        self.num_class=self.metadata['class_num']
        self.bst_rounds=[10,30,60,100]
#     @timeit
    def train(self, train_dataset, remaining_time_budget=None):
        
        """model training on train_dataset.

        :param train_dataset: tuple, (train_x, train_y)
            train_x: list of vectors, input train speech raw data.
            train_y: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        :param remaining_time_budget:
        """
        if self.use2nd_test_ths is None:
            self.use2nd_test_ths=remaining_time_budget/5
        if self.done_training:
            return
        
        if self.converter is None :
            self.converter=Converter(self.metadata['class_num'],self.metadata['train_num'],train_dataset)

        # fetching part of the train data
        if self.train_loop_num<=3:
            self.converter.fetch(self.fetch_ratio[self.train_loop_num])       
   
        self.train_loop_num += 1
        # using outer lr to fit before all train data have been fetched
        if self.use_lr :
            if self.train_loop_num <=3:
                 # continue using outer lr to fit
                data_x,data_y=self.converter.get_lr_data()
                print("now fiting outer lr at {} times".format(self.train_loop_num))
                if not self.converter.is_multitag:
                    print("multiclass using lr")
                    self.lr.fit(data_x,data_y)
                    self.acc_lr=round(0.9*self.lr.score(data_x,data_y)-0.1,6)
                else:
                    print("multilabel using lgbm")
                    params = {
                        "objective": 'binary',
                        "metric": 'auc',
                        "verbosity": -1,
                        "seed": 1
                    }
                    start=time.time()
                    bstdata=[lgb.Dataset(data_x, label=data_y[:, i], free_raw_data=False) for i in range(self.num_class)]
                    num_round=self.bst_rounds[self.train_loop_num-1]
                    if self.bst is None:
                        self.bst=[lgb.train({**params}, bstdata[i], num_boost_round=num_round,keep_training_booster=True) for i in range(self.num_class)]
                    else:
                        self.bst=[lgb.train({**params}, bstdata[i], num_boost_round=num_round,init_model=self.bst[i],keep_training_booster=True) for i in range(self.num_class)]
                    solution=np.concatenate([m.predict(data_x).reshape(-1, 1) for m in self.bst], axis=1)
                    self.acc_lr=0.98
                return
            
            # stop using lr & feed all data to engine
            self.use_lr=False
            self.train_loop_num=1
            self.use2nd=True
            self.model2nd.change_multilabel(self.converter.is_multitag)
            self.use2nd_start=time.time()

        self.model2nd.train(self.converter.get_dataset(),remaining_time_budget)
        self.done_training = self.model2nd.done_training
        print("--------------------------------------------------------------------")
        print("train done")
        print("--------------------------------------------------------------------")

#     @timeit
    def test(self, test_dataset, remaining_time_budget=None):
        """
        :param test_x: list of vectors, input test speech raw data.
        :param remaining_time_budget:
        :return: A `numpy.ndarray` matrix of shape (sample_count, class_num).
                     here `sample_count` is the number of examples in this dataset as train
                     set and `class_num` is the same as the class_num in metadata. The
                     values should be binary.
        """
        # extract test feature
        if self.observed_test_dataset is None:
            self.observed_test_dataset, _ = convert_speech(test_dataset)
            data_x=lr_pre(self.observed_test_dataset)
            scaler = StandardScaler()
            self.lr_test = scaler.fit_transform(data_x[:, :])
        test_x = self.observed_test_dataset
        
        if self.use_lr:
            # using outer lr for testing 
            if not self.converter.is_multitag:
                self.lr_last_predict=self.lr.predict(self.lr_test)
            else:
                self.lr_last_predict=np.concatenate([m.predict(self.lr_test).reshape(-1, 1) for m in self.bst], axis=1)
            return self.lr_last_predict
        
        #using 2nd model
        self.empty_times-=1

        if self.train_loop_num> 1:
            self.acc_2nd=self.model2nd.g_train_acc_list[-1]
            
        if (self.empty_times<=0) or (self.acc_2nd>self.acc_lr) or ((time.time()-self.use2nd_start)>self.use2nd_test_ths) :
            return self.model2nd.test(test_x,remaining_time_budget)
        else:
            return self.lr_last_predict


if __name__ == '__main__':
    from ingestion.dataset import AutoSpeechDataset
    D = AutoSpeechDataset(os.path.join("../sample_data/DEMO", 'DEMO.data'))
    D.read_dataset()
    m = Model(D.get_metadata())
    m.train(D.get_train())
    m.test(D.get_test())
    m.train(D.get_train())


