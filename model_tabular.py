import numpy as np
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import os
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import log_loss, roc_auc_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class tabularModel:
    def __init__(self, metadata):
        self.metadata = metadata
        self.total_train_data = metadata.size()
        self.num_feature = metadata.get_tensor_size()[1]
        self.num_class = metadata.get_output_size()
        self.session = tf.Session()
        self.done_training = False
        self.epoch = 0

        self.hyper = {
            'learning_rate': 0.01, 
            #'max_depth': 5, 
            'num_leaves': 16, 
            "min_child_samples": 5, 
            'bagging_fraction': 0.8, 
            'bagging_freq': 1, 
            "feature_fraction": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0, 
            'save_binary': True, 
            'num_threads': 4
        }
        self.min_lr = 0.0005
        self.lr_decay_ratio = 0.8
        #if (self.num_feature >= 100):
        #    self.hyper['feature_fraction'] = 0.2
        #if (self.num_feature >= 500):
        #    self.hyper['feature_fraction'] = 1. / self.num_feature

        self.iterations = np.ones(1000, dtype=int) * 50
        self.iterations[1] = 10
        self.iterations[2:4] = 20

        self.multilabel_lr = [0.01 for _ in range(self.num_class)]
        self.multilabel_done_training = np.array([False for _ in range(self.num_class)])

        self.stage = 0

        print ('feature num: ', self.num_feature)
        print ('sample num: ', self.total_train_data)
        print ('class num: ', self.num_class)
    
    def train(self, dataset, remaining_time_budget=None):
        print (self.epoch)

        # data sampling
        start = time.time()
        
        if (self.epoch == 0):
            sample_num = min(1000, self.total_train_data)
            dataset = dataset.take(sample_num)
            self.trainX, self.trainY = convert_dataset_to_numpy(dataset, sess=self.session)
            self.trainX = self.trainX.reshape(-1, self.num_feature)

            # judge output type
            if ((self.trainY.sum(axis=1) == 1).mean() == 1):
                self.output_type = 'multiclass'
            else:
                self.output_type = 'multilabel'
            print ('output type: ', self.output_type)
            print ('label ratio: ', self.trainY.mean(axis=0))

            if (self.output_type == 'multiclass'):
                self.trainY_onehot = self.trainY
                self.trainY = np.argmax(self.trainY, axis=1)

        elif (self.epoch == 6):
            sample_num = min(5000, self.total_train_data)
            dataset = dataset.take(sample_num)
            self.trainX, self.trainY = convert_dataset_to_numpy(dataset, sess=self.session)
            self.trainX = self.trainX.reshape(-1, self.num_feature)
            if (self.output_type == 'multiclass'):
                self.trainY_onehot = self.trainY
                self.trainY = np.argmax(self.trainY, axis=1)
            self.hyper['num_leaves'] = 24

        elif (self.epoch == 11):
            X, Y = convert_dataset_to_numpy(dataset, sess=self.session)
            X = X.reshape(-1, self.num_feature)
            #np.random.seed(2020)
            #index = np.arange(X.shape[0])
            #np.random.shuffle(index)
            #X, Y = X[index], Y[index]
            train_num = int(X.shape[0] * 0.8)
            self.trainX, self.validX = X[:train_num], X[train_num:]
            self.trainY, self.validY = Y[:train_num], Y[train_num:]
            if (self.output_type == 'multiclass'):
                self.trainY_onehot, self.validY_onehot = self.trainY, self.validY
                self.trainY = np.argmax(self.trainY, axis=1)
                self.validY = np.argmax(self.validY, axis=1)
            self.hyper['num_leaves'] = 32

        end = time.time()
        print ('time on loading data: %.2f' %(end - start))

        if self.output_type == 'multiclass':
            self.train_multiclass()
        else:
            self.train_multilabel()

    def train_multiclass(self):
        params = {
            'objective': 'multiclass', 
            "metric": 'multi_logloss',
            "verbosity": -1,
            "seed": 1,
            "num_class": self.num_class
        }
        self.train_data = lgb.Dataset(self.trainX, label=self.trainY, free_raw_data=False)

        # DT
        if self.epoch == 0:
            #self.model = DecisionTreeClassifier(min_samples_leaf=5, max_leaf_nodes=32).fit(self.trainX, self.trainY)
            self.model = lgb.train({**params, **self.hyper}, self.train_data, num_boost_round=1)

        # RF
        elif self.epoch == 1:
            self.model = lgb.train({**params, **self.hyper, 'boosting':'rf'}, self.train_data, num_boost_round=10)

        # GBDT fast
        elif self.epoch == 2:
            self.model = lgb.train({**params, **self.hyper}, self.train_data, num_boost_round=20, keep_training_booster=True)
        elif (self.epoch < 6):
            self.model = lgb.train({**params, **self.hyper}, self.train_data, num_boost_round=20, 
                init_model=self.model, keep_training_booster=True)

        # GBDT mid
        elif (self.epoch == 6):
            self.model = lgb.train({**params, **self.hyper}, self.train_data, num_boost_round=20, keep_training_booster=True)
        elif (self.epoch < 11):
            self.model = lgb.train({**params, **self.hyper}, self.train_data, num_boost_round=20, 
                init_model=self.model, keep_training_booster=True)
        
        # GBDT full
        elif (self.epoch == 11):
            self.model = lgb.train({**params, **self.hyper}, self.train_data, num_boost_round=100, keep_training_booster=True)
            valid_score = roc_auc_score(self.validY_onehot, self.model.predict(self.validX), average='macro')
            print ('valid score: ', valid_score)
            self.best_score = valid_score
        else:
            self.new_model = lgb.train({**params, **self.hyper}, self.train_data, num_boost_round=50, 
                init_model=self.model, keep_training_booster=True)
            # decrease learning rate if the performance does not improve
            #valid_score = log_loss(self.validY, new_model.predict(self.validX))
            valid_score = roc_auc_score(self.validY_onehot, self.new_model.predict(self.validX), average='macro')
            print ('valid score: ', valid_score)
            if (valid_score > self.best_score):
                self.best_score = valid_score
                self.model = self.new_model
            else:
                self.done_training = True
                return

    def train_multilabel(self):
        params = {
            "objective": 'binary',
            "metric": 'auc',
            "verbosity": -1,
            "seed": 1
        }
        self.train_data = [lgb.Dataset(self.trainX, label=self.trainY[:, i], free_raw_data=False) for i in range(self.num_class)]

        # DT
        if self.epoch == 0:
            self.model = [lgb.train({**params, **self.hyper}, self.train_data[i], num_boost_round=1) for i in range(self.num_class)]
        
        # RF
        elif self.epoch == 1:
            self.model = [lgb.train({**params, **self.hyper, 'boosting':'rf'}, self.train_data[i], num_boost_round=10) for i in range(self.num_class)]
        
        # GBDT fast
        elif self.epoch == 2:
            self.model = [lgb.train({**params, **self.hyper}, self.train_data[i], num_boost_round=20, 
                keep_training_booster=True) for i in range(self.num_class)]
        elif self.epoch < 6:
            self.model = [lgb.train({**params, **self.hyper}, self.train_data[i], num_boost_round=20, 
                init_model=self.model, keep_training_booster=True) for i in range(self.num_class)]

        # GBDT mid
        elif self.epoch == 6:
            self.model = [lgb.train({**params, **self.hyper}, self.train_data[i], num_boost_round=20, 
                keep_training_booster=True) for i in range(self.num_class)]
        elif self.epoch < 11:
            self.model = [lgb.train({**params, **self.hyper}, self.train_data[i], num_boost_round=20, 
                init_model=self.model, keep_training_booster=True) for i in range(self.num_class)]

        # GBDT full
        elif (self.epoch == 11):
            self.model = [lgb.train({**params, **self.hyper}, self.train_data[i], num_boost_round=100, 
                keep_training_booster=True) for i in range(self.num_class)]
            self.best_score = [roc_auc_score(self.validY[:, i], self.model[i].predict(self.validX), average='macro') for i in range(self.num_class)]
        else:
            for i in range(self.num_class):
                if self.multilabel_done_training[i]:
                    continue
                print ('class ', i)
                new_model = lgb.train({**params, **self.hyper}, self.train_data[i], num_boost_round=50, 
                    init_model=self.model[i], keep_training_booster=True)
                valid_score = roc_auc_score(self.validY[:, i], new_model.predict(self.validX))
                print ('valid score: ', valid_score)
                if (valid_score > self.best_score[i]):
                    self.best_score[i] = valid_score
                    self.model[i] = new_model
                else:
                    self.multilabel_done_training[i] = True

            if (self.multilabel_done_training.mean() == 1):
                self.done_training = True
                return

    def test(self, dataset, remaining_time_budget=None):
        """
        :param: Same as that of `train` method, except that the labes will be empty (all zeros)
        :return: predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim)
            The values should be binary or in the interval [0,1].
        """
        if (self.epoch == 0):
            test_data, label = convert_dataset_to_numpy(dataset, sess=self.session)
            self.testX = np.reshape(test_data, (-1, self.num_feature))

        if (self.output_type == 'multiclass'):
            prediction = self.model.predict(self.testX)
        else:
            prediction = np.concatenate([m.predict(self.testX).reshape(-1, 1) for m in self.model], axis=1)

        self.epoch += 1

        return prediction


def convert_dataset_to_numpy(dataset, batch=512, sess=tf.Session()):
    dataset = dataset.batch(batch)
    values = []
    labels = []
    next_ele = dataset.make_one_shot_iterator().get_next()
    #values, labels = sess.run(next_ele)
    
    while 1:
        try:
            value, label = sess.run(next_ele)
            values.append(value)
            labels.append(label)
        except:
            # means we reach the bottom
            break
    values = np.concatenate(values, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return values, labels