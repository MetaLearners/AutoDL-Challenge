import torch
import os
import time
from .nlp_models.svm import svmModel
from .nlp_models.kerasCnnModel import kerasCnnModel
from sklearn.feature_extraction.text import CountVectorizer
from utils.models.majorClassModel import majorClassModel as majorModel
from .data_manager import DataPipeline, DataProcessor
from utils.score import autodl_auc
from .data_processors import Idx2SIdx
import numpy as np
import gzip
import pickle

SAMPLE_NUM = 5000

class ModelZoo(object):
    """
    Manage all the nlp models we use
    """
    def __init__(self, metadata, timer, is_ZH, data_manager, model_list=['bert','svm','keras'], use_pretrain=True, sess=None, global_model=None):
        super().__init__()
        self.timer = timer
        self.global_model = global_model
        self.metadata = metadata
        self.data = data_manager
        self.num_class = self.data.num_class
        self.is_ZH = is_ZH
        self.timer("model zoo init")
        self.models = model_list
        self.use_pretrain = use_pretrain
        self._preprocess_meta()
        need_emb = ['keras']
        need_emb = [(x in self.models) for x in need_emb]
        if sum(need_emb) > 0:
            self.use_filter = True
            self._load_emb()
            self.max_vocab_size = 20000
            self.shrink_gen = False
            self.cur_vocab_size = self.pretrained_emb.shape[0]
            self.use_shrink_emb = self.max_vocab_size < self.cur_vocab_size
            if self.use_shrink_emb:
                if self.use_filter:
                    self.re_index_shrink_emb = [x for x in self.re_index_lower_filter]
                else:
                    self.re_index_shrink_emb = [self.max_vocab_size - 2] * len(self.vocab)
                print('[zoo] detect total %d words, will use shrinked embedding' % (self.cur_vocab_size))
            else:
                print('[zoo] detect total %d words, it\'s still safe to use' % (self.cur_vocab_size))
        if 'svm' in self.models:
            self.svm = svmModel(timer, metadata, is_ZH, data_manager, self)
        if 'bert' in self.models:
            from .nlp_models.bertModel import bertModel
            self.bert = bertModel(metadata, timer, is_ZH, data_manager)
        if 'keras' in self.models:
            self.keras = kerasCnnModel(metadata, data_manager, self.fasttext_embedding, self.metadata.get_output_size(), sess, is_ZH)
        self.valid_label = None
        self.timer("model zoo init")
        self.valid_score = []
        self.valid_model = []
        print("[model zoo] init. time cost: %.2f s" % (self.timer.accumulation["model zoo init"]))

    def train(self, index=None, model='svm', skip_train=False, skip_valid=False, epoch_key=-1):
        if model == 'svm':
            if not skip_train:
                print('[zoo train] current model svm. begin training ...')
                self.svm.train(index, epoch_key=epoch_key)
            if not skip_valid:
                res = self.svm.valid()
                if self.valid_label is None:
                    self.valid_label = self.data.valid_label_all
                self.timer("eval-score")
                score = autodl_auc(self.valid_label, res)
                delta = self.timer("eval-score")
                print('[zoo] calculate score time cost: %.2f s' % (delta))
                self.valid_score.append(score)
                self.valid_model.append('svm')
        elif model == 'bert':
            if not skip_train:
                print('[zoo train] current model bert. begin training ...')
                self.bert.train(index, epoch_key=epoch_key)
            if not skip_valid:
                res = self.bert.valid()
                if self.valid_label is None:
                    self.valid_label = self.data.valid_label_all
                self.timer("eval-score")
                score = autodl_auc(self.valid_label, res)
                delta = self.timer("eval-score")
                print('[zoo] calculate score time cost: %.2f s' % (delta))
                self.valid_score.append(score)
                self.valid_model.append('bert')
        elif model == 'kerasCnn':
            if not skip_train:
                print('[zoo train] current model keras-cnn. begin training ...')
                self.keras.train(index, epoch_key=epoch_key)
            if not skip_valid:
                res = self.keras.valid()
                if self.valid_label is None:
                    self.valid_label = self.data.valid_label_all
                score = autodl_auc(self.valid_label, res)
                self.valid_score.append(score)
                self.valid_model.append('kerasCnn')
        else:
            raise KeyError("model %s not found!" % (model))

    def predict(self, model='svm'):
        if model == 'svm':
            print('[zoo test] current model svm, begin testing ...')
            return self.svm.predict()
        elif model == 'bert':
            print('[zoo test] current model bert, begin testing ...')
            return self.bert.test()
        elif model == 'kerasCnn':
            print('[zoo test] current model keras-cnn. begin testing ...')
            return self.keras.predict()
        else:
            raise KeyError("model %s not found!" % (model))
    
    def _preprocess_meta(self):
        # here, we want to construct a re_index to cast origin idx to processed idx
        pattern = '[“”【】/（）：「」、|，；。"/\\(){}[]|@,;]."#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n'
        filter_word = set(pattern)

        vocab = self.metadata.get_channel_to_index_map()
        self.vocab = vocab

        self.re_index_lower_filter = [-1] * len(vocab)
        new_vocab = {}
        new_vocab_list = []
        filtered_word = []
        idx = 0
        for v in vocab:
            if v in filter_word:
                filtered_word.append(v)
                continue
            vl = v.lower()
            if vl in new_vocab:
                self.re_index_lower_filter[vocab[v]] = new_vocab[vl]
                new_vocab_list[new_vocab[vl]].append(vocab[v])
            else:
                self.re_index_lower_filter[vocab[v]] = idx
                new_vocab[vl] = idx
                new_vocab_list.append([vocab[v]])
                idx += 1

        self.new_vocab_to_idx = new_vocab
        self.new_vocab_idx_map = new_vocab_list

        print('[model manager] new vocab size: %d, origin vocab size: %d, ignoring words: %s' % (len(new_vocab), len(vocab), str(filtered_word)))

        self.data.add_pipeline(DataPipeline('idx2idxLowerPipe', pipeline=[
            Idx2SIdx(None, 'idx2sidxLower')
        ]))

        self.data.processor_pool['idx2sidxLower'].re_index = np.array(self.re_index_lower_filter)

    def _load_emb(self, random=False):
        # loading pretrained embedding
        time1 = time.time()
        FT_DIR = '/app/embedding'
        if not os.path.exists(FT_DIR):
            FT_DIR = '/DATA-NFS/guancy/autoDL/app/embedding'
        
        ft_word2idx = {}
        ft_emb = []
        self.fasttext_embedding = {}
        if os.path.exists('ft_word2idx.bin') and self.use_pretrain:
            ft_word2idx = pickle.load(open('ft_word2idx.bin', 'rb'))
            ft_emb = pickle.load(open('ft_emb.bin', 'rb'))
            self.fasttext_embedding = pickle.load(open('fasttext_embedding.bin', 'rb'))
            mean_emb = np.mean(ft_emb, 0)
            std_emb = np.std(ft_emb, 0)
        elif not random and self.use_pretrain:
            if self.is_ZH:
                f = gzip.open(os.path.join(FT_DIR, 'cc.zh.300.vec.gz'), 'rb')
            else:
                f = gzip.open(os.path.join(FT_DIR, 'cc.en.300.vec.gz'), 'rb')

            curIdx = 0
            for line in f.readlines()[1:]:
                values = line.strip().split()
                if self.is_ZH:
                    word = values[0].decode('utf8')
                else:
                    word = values[0].decode('utf8')
                if len(values) != 301:
                    print('[warn] word: %s does not have 300 dim. Only %d get' % (word, len(values) - 1))
                    continue
                ft_word2idx[word] = curIdx
                curIdx += 1
                coefs = np.asarray(values[1:], dtype='float32')
                self.fasttext_embedding[word] = coefs
                ft_emb.append(coefs)
            ft_emb = np.array(ft_emb)
            mean_emb = np.mean(ft_emb, 0)
            std_emb = np.std(ft_emb, 0)
            print('Found %s fastText word vectors.' % curIdx)
            time2 = time.time()
            print('[init] load pretrained emb. time cost: %.2f s' % (time2 - time1))
            pickle.dump(ft_word2idx, open('ft_word2idx.bin', 'wb'))
            pickle.dump(ft_emb, open('ft_emb.bin', 'wb'))
            pickle.dump(self.fasttext_embedding, open('fasttext_embedding.bin', 'wb'))
        else:
            print('[zoo] DO NOT USE PRETRAINED EMBEDDING!')
            ft_emb = np.random.randn(20, 300)
            mean_emb = np.mean(ft_emb, 0)
            std_emb = np.std(ft_emb, 0)

            
        # re-index
        # we need to clean the text here also
        # we also need to lowercase all the words
        vocab = self.vocab

        if self.use_filter:
            _pretrained_list = [None] * len(self.new_vocab_to_idx)
            oov = 0
            for v in self.new_vocab_to_idx:
                if v in ft_word2idx:
                    _pretrained_list[self.new_vocab_to_idx[v]] = ft_emb[ft_word2idx[v]]
                else:
                    oov += 1
                    _pretrained_list[self.new_vocab_to_idx[v]] = mean_emb + np.random.randn(ft_emb.shape[1]) * std_emb
                
            # add <pad>
            _pretrained_list.append(np.zeros(300))

            self.pretrained_emb = np.array(_pretrained_list)
            print('[init pretrained emb] oov / total: %d / %d' % (oov, self.pretrained_emb.shape[0]))

        else:

            self.pretrained_emb = np.zeros((len(vocab) + 1, ft_emb.shape[1]))
            index_to_token = [None] * len(vocab)
            for token in vocab:
                index = vocab[token]
                index_to_token[index] = token
            oov = 0
            for i in range(len(vocab)):
                if index_to_token[i] in ft_word2idx:
                    self.pretrained_emb[i] = ft_emb[ft_word2idx[index_to_token[i]]]
                else:
                    oov += 1
                    self.pretrained_emb[i] = mean_emb + np.random.randn(ft_emb.shape[1]) * std_emb
            self.pretrained_emb[-1] = np.zeros(300)
            print('[init] oov word / total word: %d / %d' % (oov, self.pretrained_emb.shape[0]))
    
    def _shrink_pretrained_emb(self):
        if self.shrink_gen:
            return
        print('[zoo] shrink emb begin!')
        time1 = time.time()
        
        # <unk> index: self.max_vocab_size - 2
        # <pad> index: self.max_vocab_size - 1

        if self.use_filter:
            self.data.reset()
            sampled_x = self.data.get_batch('idx2idxLowerPipe', self.data.get_rounded_batch_size(SAMPLE_NUM))[0]
            # count index in sampled x
            count_vec = [0] * len(self.new_vocab_idx_map)
            for line in sampled_x:
                for i in line:
                    count_vec[i] += 1

            sort_vec = sorted(zip(count_vec, self.new_vocab_idx_map, range(len(self.new_vocab_idx_map))), key=lambda x:-x[0])

            # build a reverse re-index
            shrinked_emb_idx = []
            for idx in range(len(self.new_vocab_idx_map)):
                cur_word_origin_idx = sort_vec[idx][1]
                if idx < self.max_vocab_size - 2:
                    # we keep them pointing to valid idx
                    for oi in cur_word_origin_idx:
                        self.re_index_shrink_emb[oi] = idx
                    shrinked_emb_idx.append(sort_vec[idx][2])
                else:
                    # we point them to <unk>
                    for oi in cur_word_origin_idx:
                        self.re_index_shrink_emb[oi] = self.max_vocab_size - 2
            shrinked_emb = self.pretrained_emb[shrinked_emb_idx]
            #re_index = self.re_index_shrink_emb
            #re_index[-1] = shrinked_emb.shape[0] + 1
        else:
            # count index in sampled x
            sample_per_class = min([len(x) for x in self.data.train_data])
            max_sample = SAMPLE_NUM // self.data.num_class
            sampled = min([max_sample, sample_per_class])
            sampled_x = []
            for i in range(self.data.num_class):
                sampled_x.extend(np.random.choice(self.data.train_data[i], (sampled, )).tolist())
            count_vec = [0] * len(self.vocab)
            for line in sampled_x:
                for i in line:
                    count_vec[i] += 1
        
            sort_vec = sorted(zip(count_vec, range(len(self.vocab))), key=lambda x:-x[0])

            for idx in range(len(self.vocab)):
                if idx < self.max_vocab_size - 2:
                    self.re_index_shrink_emb[sort_vec[idx][1]] = idx
                else:
                    break
            #re_index[-1] = self.max_vocab_size - 1
            shrinked_emb = self.pretrained_emb[list(map(lambda x:x[1], sort_vec[:self.max_vocab_size - 2]))]

        listed = shrinked_emb.tolist()
        listed.append(np.random.randn(300) * 0.06)
        listed.append(np.zeros(300))
        self.shrinked_emb = np.array(listed)

        self.data.add_pipeline(DataPipeline('idx2sidxPipe', pipeline=[
            Idx2SIdx(None, 'idx2sidx'),
        ]))

        self.data.processor_pool['idx2sidx'].re_index = np.array(self.re_index_shrink_emb)

        self.shrink_gen = True

        time2 = time.time()
        print('[zoo] shrink embedding ... time cost: %.2f s' % (time2 - time1))

    def get_max_lower_length(self):
        if hasattr(self, 'lower_length'):
            return self.lower_length
        self.data.reset()
        sent = self.data.get_batch('idx2idxLowerPipe', self.data.get_rounded_batch_size(1000))
        self.data.reset()
        max_length = max([len(x) for x in sent[0]])
        print('[model zoo] max lower length: %d' % (max_length))
        self.lower_length = max_length
        return max_length
    
    def get_max_shrink_length(self):
        if hasattr(self, 'shrink_length'):
            return self.shrink_length
        self.data.reset()
        sent = self.data.get_batch('idx2sidxPipe', self.data.get_rounded_batch_size(1000))
        self.data.reset()
        max_length = max([len(x) for x in sent[0]])
        print('[model zoo] max shrink length: %d' % (max_length))
        self.shrink_length = max_length
        return max_length