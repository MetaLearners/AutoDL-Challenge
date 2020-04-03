"""
model nlp ver 25

1. deal with Chinese
2. deal with multi-label
"""

import os
from nlp_utils.timer import Timer, stampTimer

GLOBAL_TIMER = stampTimer()
import tensorflow as tf
import zipfile
import copy
import json
import torch
import time
import random
import numpy as np
from nlp_utils.model_zoo import ModelZoo
from nlp_utils.data_manager import nlp_dataset as Dataset
from nlp_utils.CONSTANT import *

import keras
import keras.backend as K

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_chinese(metadata):
  """Judge if the dataset is a Chinese NLP dataset. The current criterion is if
  each word in the vocabulary contains one single character, because when the
  documents are in Chinese, we tokenize each character when formatting the
  dataset.

  Args:
    metadata: an AutoDLMetadata object.
  """

  for i, token in enumerate(metadata.get_channel_to_index_map()):
    if len(token) != 1:
      return False
    if i >= 100:
      break
  return True

class nlpModel:
    def __init__(self, metadata):
        super().__init__()
        GLOBAL_TIMER("init nlp model")
        print('[init nlp model] begin initializing nlp models')
        set_seed(1)
        self.metadata = metadata
        self.is_ZH = is_chinese(metadata)
        print("[init nlp model] initialize dataset")
        GLOBAL_TIMER("init dataset")

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.global_sess = sess
        self.data = Dataset(metadata, self.global_sess)
        K.set_session(self.global_sess)
        GLOBAL_TIMER("init dataset")
        print("[init nlp model] initialize dataset done. time cost: %.2f s" % (GLOBAL_TIMER.get_latest_delta("init dataset")))
        self.zoo = ModelZoo(metadata, GLOBAL_TIMER, is_ZH=self.is_ZH, data_manager=self.data, model_list=['svm', 'keras', 'bert'], use_pretrain=True, sess=self.global_sess, global_model=self)
        self.train_data = None
        self.test_data = None
        print('[init nlp model] language', '----:::: ENGLISH ::::----' if not self.is_ZH else '----:::: CHINESE ::::----')
        self.epoch = 0
        self.full_svm = False
        self.same_patience = 5
        self.diff_patience = 10
        self.model_sequence = ['svm', 'kerasCnn', 'bert']
        self.multi_length_mode = [500, 1000, 2500, 5000]
        self.multi_batch_size = {
            'kerasCnn': [256, 128, 64, 32],
            'fusionCnn': [256, 128, 64, 32],
            'cnn': [256, 128, 64, 32],
        }
        self.model_idx = 0
        self.last_idx = 0
        self.test_model = 'None'
        self.done_training = False
        self.change_model = False
        self.do_test = False
        self.max_input_length = -1
        self.cur_length_id = 0
        self.do_ensemble = False
        if self.do_ensemble:
            self.result_list = {}

        GLOBAL_TIMER("init nlp model")
        print("[init nlp model] initializing nlp model done. %.2f s" % (GLOBAL_TIMER.get_latest_delta("init nlp model")))

    def setup_length(self):
        cut_idx = sum(np.array(self.multi_length_mode) < self.max_input_length)
        if cut_idx < 4:
            self.multi_length_mode = self.multi_length_mode[:cut_idx] + [self.max_input_length]
            for k in self.multi_batch_size:
                self.multi_batch_size[k] = self.multi_batch_size[k][:cut_idx + 1]
        print(self.multi_length_mode)
        print(self.multi_batch_size)
        
        if 'kerasCnn' in self.model_sequence:
            self.zoo.keras.set_multi_length(self.multi_length_mode)
        '''
        if 'cnn' in self.model_sequence:
            self.zoo.cnn.set_multi_length(self.multi_length_mode)
        '''
        
    def train(self, dataset, remaining_time_budget=None):
        GLOBAL_TIMER("TRAIN")
        # automatically determine model
        time1 = time.time()
        if self.model_idx >= len(self.model_sequence):
            self.done_training = True
            return
        model = self.model_sequence[self.model_idx]
        self.test_model = model
        print('[train] model: %s, epoch: %d. BEGIN' % (model, self.epoch))

        if self.data.train_data_done == False:
            self.data.set_train(dataset)

        if self.epoch == 0:
            # svm
            self.zoo.train(index=get_proper_data(self.data.num_class), model='svm', skip_train=False, skip_valid=True, epoch_key=self.epoch)
            self.model_idx = 0
            self.test_model = 'svm'
            self.data.reset()
            print('[first svm train done] TIME USED NOW %.2f s' % (get_total_time()))
            self.epoch += 1
            return

        if self.epoch == 1:
            # we need to first get the valid info of last turn
            self.data.get_meta_info()
            max_length = self.data.max_length
            self.max_input_length = max_length
            self.setup_length()
            avg_length = self.data.avg_length
            char_num = len(self.zoo.pretrained_emb)
            print('[META] max length: %d, avg length: %d, vocab size: %d' % (max_length, avg_length, char_num))
            print('[epoch 1 valid before] TIME USED NOW %.2f s' % (get_total_time()))
            self.zoo.train(index=0, model=self.test_model, skip_train=True, skip_valid=False, epoch_key=self.epoch)
            self.canWeStop()
            print('[epoch 1 valid after] TIME USED NOW %.2f s' % (get_total_time()))
            self.epoch += 1
            time2 = time.time()
            self.model_idx = 1
            self.train(dataset, remaining_time_budget=remaining_time_budget - time2 + time1)
            return

        if model == 'svm':
            self.zoo.train(index=get_proper_data(self.data.num_class), model='svm', skip_train=self.epoch==1, skip_valid=self.epoch==0, epoch_key=self.epoch)
            self.epoch += 1
            if self.epoch == 2:
                self.data.reset()
                self.model_idx += 1
                time2 = time.time()
                self.train(dataset, remaining_time_budget=remaining_time_budget - time2 + time1)
                return

        elif model == 'bert':
            #if self.zoo.bert.place == 'cpu':
            #    self.zoo.bert.to_deivce('cuda')
            canStop = False
            if self.zoo.bert.estimate_time_per_batch is None:
                self.zoo.bert.estimate_time_per_batch = 1
            if self.zoo.bert.estimate_valid_time is None:
                self.zoo.bert.estimate_valid_time = self.data.valid_count / self.zoo.bert.batch_size_eval * self.zoo.bert.estimate_time_per_batch
            if self.zoo.bert.estimate_test_time is None:
                self.zoo.bert.estimate_test_time = self.data.test_count / self.zoo.bert.batch_size_eval * self.zoo.bert.estimate_time_per_batch
            while not self.done_training and not canStop:
                self.zoo.bert.batch_per_train = min([self.zoo.bert.batch_per_train, self.E_max_epoch('bert')])
                print('[model] estimate max epoch is : %d' % (self.E_max_epoch('bert')))
                if self.zoo.bert.batch_per_train <= 0:
                    self.done_training = True
                    self.change_model = False
                if not self.done_training:
                    self.zoo.train(index=self.data.get_rounded_batch_size(self.zoo.bert.batch_size), model='bert', epoch_key=self.epoch)
                    self.epoch += 1
                    canStop = self.canWeStop()
            if self.change_model and not self.done_training:
                self.change_model = False
                self.data.reset()
                self.model_idx += 1
                #self.zoo.bert.to_deivce('cpu')
                time2 = time.time()
                self.train(dataset, remaining_time_budget=remaining_time_budget - time2 + time1)
        
        elif model in ['kerasCnn']:
            canStop = False
            mod = {
                'kerasCnn': None if not hasattr(self.zoo, 'keras') else self.zoo.keras,
            }
            if mod[model].estimate_time_per_batch is None:
                mod[model].estimate_time_per_batch = 20
            if mod[model].estimate_valid_time is None:
                mod[model].estimate_valid_time = 20
            if mod[model].estimate_test_time is None:
                mod[model].estimate_test_time = 20
            while not self.done_training and not canStop:
                if 1200 - time.time() + GLOBAL_TIMER.accumulation_time['TRAIN'][0] - mod[model].estimate_time_per_batch - mod[model].estimate_valid_time - mod[model].estimate_test_time - 20 < 0:
                    self.done_training = True
                    self.change_model = False
                if not self.done_training:
                    print('[%s train before] TIME USED NOW %.2f s' % (model, get_total_time()))
                    #self.zoo.train(index=get_proper_data(self.data.num_class), model=model, epoch_key=self.epoch)
                    self.zoo.train(index=get_rounded_batch_size(128 * 40, self.data.num_class), model=model, epoch_key=self.epoch)
                    print('[%s train after] TIME USED NOW %.2f s' % (model, get_total_time()))
                    self.epoch += 1
                    canStop = self.canWeStop()
            if self.change_model and not self.done_training:
                #mod[model].clear_gpu()
                self.change_model = False
                self.data.reset()
                self.model_idx += 1
                time2 = time.time()
                self.train(dataset, remaining_time_budget=remaining_time_budget - time2 + time1)

        else:
            raise ValueError('no model named %s detected' % (model))

    def canWeStop(self):
        
        if len(self.zoo.valid_model) > 0:
            # manage the lr scheduler
            if self.zoo.valid_model[-1] == 'kerasCnn':
                self.zoo.valid_model[-1] = 'kerasCnn_%d' % (self.multi_length_mode[self.cur_length_id])
                if self.zoo.keras.scheduler.step(self.zoo.valid_score[-1]):
                    print('[judge] CHANGE LR OF KERAS CNN TO %f' % (self.zoo.keras.scheduler.lr))
                    K.set_value(self.zoo.keras.model.optimizer.lr, self.zoo.keras.scheduler.lr)
            print('[judge] current valid score: %.4f' % (self.zoo.valid_score[-1]))

        if len(self.zoo.valid_score) in [0, 1]:
            print('[judge] no valid or first epoch, will skip')
            return True

        model = self.zoo.valid_model[-1]

        maps = {
            'kerasCnn': None if 'kerasCnn' not in self.model_sequence else self.zoo.keras,
        }

        for key in maps:

            if model.startswith(key):
                current_model_begin_idx = self.zoo.valid_model.index('%s_%d' % (key, self.multi_length_mode[0]))
                current_state_begin_idx = self.zoo.valid_model.index(model)
                current_epoch_idx = len(self.zoo.valid_model) - 1
                max_score_idx = np.argmax(self.zoo.valid_score)
                max_score_current_model_idx = np.argmax(self.zoo.valid_score[current_model_begin_idx:]) + current_model_begin_idx
                max_score_current_model = self.zoo.valid_score[max_score_current_model_idx]
                # >>> when current state do not improve current models best score for too long time <<<
                if current_epoch_idx == max_score_idx:
                    print('[judge] ADVANCED RESULT FOR TEST')
                    return True
                if current_epoch_idx - max_score_current_model_idx > 5 and current_epoch_idx - current_state_begin_idx > 3:
                    print('[judge] LENGTH %d DO NOT IMPROVE FOR TOO LONG' % (self.multi_length_mode[self.cur_length_id]))
                    self.cur_length_id += 1
                    if self.cur_length_id >= len(self.multi_length_mode):
                        print('[judge] CHANGE MODEL')
                        self.cur_length_id = 0
                        self.change_model = True
                        return True
                    else:
                        # update models
                        maps[key].cur_length = self.multi_length_mode[self.cur_length_id]
                        maps[key].batch_size = self.multi_batch_size[key][self.cur_length_id]
                        maps[key].estimate_time_per_batch *= 2
                        maps[key].estimate_test_time *= 2
                        maps[key].estimate_valid_time *= 2
                        print('[judge] CHANGE STATE TO %d' % (self.multi_length_mode[self.cur_length_id]))
                        self.data.reset()
                        return False
                else:
                    print('[judge] CUR LENGTH %d' % (self.multi_length_mode[self.cur_length_id]))
                return False

        current_model_begin_idx = self.zoo.valid_model.index(model)
        current_state_begin_idx = current_model_begin_idx
        max_score_idx = np.argmax(self.zoo.valid_score)
        max_score = self.zoo.valid_score[max_score_idx]
        max_score_current_model_idx = np.argmax(self.zoo.valid_score[current_model_begin_idx:]) + current_model_begin_idx
        max_score_current_model = self.zoo.valid_score[max_score_current_model_idx]


        # >>> when the score is high enough, will exit training <<<
        # >>> we do not use this credit <<<
        if max_score > 0.995:
            print('[judge] the score on valid is too high: %.4f, will exit' % (max_score))
            self.done_training = True
            self.do_test = True
            return True
        
        # >>> when the score is improved, will go on testing <<<
        if max_score_idx == len(self.zoo.valid_score) - 1:
            # the last turn is the max
            print('[judge] advance result received, will break for test')
            return True
        
        # >>> when the score of newest model is not big enough to outperform previous model <<<
        if max_score_idx < current_model_begin_idx:
            if len(self.zoo.valid_score) - max_score_current_model_idx > self.same_patience:
                #if len(self.zoo.valid_score) - begin_idx > self.diff_patience:
                print('[judge] score of model %s did not improve itself for too long, will change model' % (model))
                self.change_model = True
                return True
            print('[judge] score of model %s did not improve previous result, diff %.4f, continue training'
                % (model, max_score - self.zoo.valid_score[-1]))
            # >>> when the diff is larger than 10%, we double model's train epoch
            if max_score - max(self.zoo.valid_score[current_model_begin_idx:]) > 0.1:
                print('[judge] the diff is larger than 0.1, will double train epoch')
                if model == 'bert':
                    self.zoo.bert.batch_per_train *= 2
                    self.zoo.bert.batch_per_train = min([self.zoo.bert.batch_per_train, 400])
            return False
            
        # >>> when the score don't improve for self.patience epochs, will exit training <<<
        if len(self.zoo.valid_score) - max_score_idx - 1 > self.same_patience:
            print('[judge] score didn\'t improve for %d epochs, will change model' % (self.same_patience + 1))
            self.change_model = True
            return True

        print('[judge] score didn\'t improve for %d epochs, continue training' % (len(self.zoo.valid_score) - max_score_idx - 1))
        return False

    def test(self, dataset, remaining_time_budget=None):
        if self.data.test_data_done == False:
            self.data.set_test(dataset)
        if self.done_training and not self.do_test:
            return None
        self.do_test = False
        GLOBAL_TIMER("test")
        res = self.zoo.predict(model=self.test_model)
        GLOBAL_TIMER("test")
        print("[test] time cost: %.2f s" % (GLOBAL_TIMER.get_latest_delta("test")))
        return res

    def E_max_epoch(self, model):
        if model == 'bert':
            remaining = 20 * 60 - (time.time() - GLOBAL_TIMER.accumulation_time['TRAIN'][0])
            time_to_train = remaining - self.zoo.bert.estimate_valid_time - self.zoo.bert.estimate_test_time - 60
            return time_to_train // self.zoo.bert.estimate_time_per_batch

def get_total_time():
    return time.time() - GLOBAL_TIMER.accumulation_time['TRAIN'][0]