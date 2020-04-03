"""
Dataset manager of nlp task, batch version

Designed to automatically transform the content to the wanted form.
"""
import numpy as np
import tensorflow as tf
import os
import time
import json
os.system("pip install jieba_fast")
import jieba_fast as jieba
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy.sparse
from scipy.sparse import vstack
import copy
import re
from .CONSTANT import *

def clean_en_text(dat, ratio=0.1, is_ratio=True):

    REPLACE_BY_SPACE_RE = re.compile('["/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-zA-Z #+_]')

    ret = []
    for line in dat:
        # text = text.lower() # lowercase text
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        # line = BAD_SYMBOLS_RE.sub('', line)
        line = line.strip()
        line_split = line.split()

        if is_ratio:
            NUM_WORD = max(int(len(line_split) * ratio), MAX_SEQ_LENGTH)
        else:
            NUM_WORD = MAX_SEQ_LENGTH

        if len(line_split) > NUM_WORD:
            line = " ".join(line_split[0:NUM_WORD])
        ret.append(line)
    return ret


def clean_zh_text(dat, ratio=0.1, is_ratio=False):
    REPLACE_BY_SPACE_RE = re.compile('[“”【】/（）：！～「」、|，；。"/(){}\[\]\|@,\.;]')

    ret = []
    for line in dat:
        line = REPLACE_BY_SPACE_RE.sub(' ', line)
        line = line.strip()

        if is_ratio:
            NUM_CHAR = max(int(len(line) * ratio), MAX_CHAR_LENGTH)
        else:
            NUM_CHAR = MAX_CHAR_LENGTH

        if len(line) > NUM_CHAR:
            # line = " ".join(line.split()[0:MAX_CHAR_LENGTH])
            line = line[0:NUM_CHAR]
        ret.append(line)
    return ret

def _tokenize_chinese_words(text):
    return ' '.join(jieba.cut(text, cut_all=False))

def vectorize_data(x, vectorizer=None, algo='tfidf'):
    if algo == 'tfidf':
        if vectorizer is None:
            vectorizer = TfidfVectorizer(ngram_range=(1, 1))
            vectorizer.fit(x)
    else:
        if vectorizer is None:
            vectorizer = CountVectorizer(ngram_range=(1, 1))
            vectorizer.fit(x)
    vectorized = vectorizer.transform(x)
    return vectorized, vectorizer

def concat_result(r_list):
    while r_list.count(None) > 0:
        r_list.remove(None)
    if len(r_list) == 1:
        return r_list[0]
    if len(r_list) == 0:
        return None
    return [concat_data([a[i] for a in r_list]) for i in range(len(r_list[0]))]

def concat_data(d_list):
    if isinstance(d_list[0], np.ndarray):
        return np.concatenate(d_list)
    if isinstance(d_list[0], list):
        res = []
        for d in d_list:
            res.extend(d)
        return res
    if scipy.sparse.issparse(d_list[0]):
        return vstack(d_list)
    raise TypeError("cannot concat following type: " % (str([str(type(d)) for d in d_list])))


class nlp_dataset:
    def __init__(self, metadata, global_sess):
        self.multi_label_mode = False
        self.metadata = metadata
        self.num_class = int(self.metadata.get_output_size())
        self.train_data = [[] for _ in range(self.num_class)]
        self.train_label = [[] for _ in range(self.num_class)]
        self.test_data = []
        self.valid_data = [[] for _ in range(self.num_class)]
        self.valid_label = [[] for _ in range(self.num_class)]
        self.valid_label_all = []
        self.valid_batch = get_valid_per_sample(self.num_class)
        self.train_count = 0
        self.valid_count = 0
        self.test_count = 0
        self.train_data_done = False
        self.test_data_done = False
        self.processor_pool = {}
        self.pipeline_pool = {}
        self.processor_data = {'train':[{} for _ in range(self.num_class)], 'valid':[{} for _ in range(self.num_class)], 'test':{}}
        self.sess = global_sess
        self.dataset = None
        self.train_idx = [0] * self.num_class
        self.total_num = -1
        self.dataset_exhausted = False
        self.extend_number = 1000
    
    def update_valid_label(self):
        if len(self.valid_label_all) == 0:
            self.valid_label_all = []
            for i in range(self.num_class):
                self.valid_label_all.extend(self.valid_label[i])
            self.valid_label_all = np.array(self.valid_label_all)

    def get_max_length(self):
        return self.max_length
        
    def set_train(self, train):
        self.train = train
        self.train_next = self.train.make_one_shot_iterator().get_next()
        self.train_data_done = True
    
    def set_test(self, test):
        self.test = test
        self.test_next = self.test.make_one_shot_iterator().get_next()
        self.test_data_done = True

    def migrate_to_multi_label_mode(self):
        # used to transform current data to multilabel mode
        pass

    def add_pipeline(self, pipeline):
        if pipeline.name not in self.pipeline_pool:
            self.pipeline_pool[pipeline.name] = [_get_pipeline_name(p) for p in pipeline.pipeline]
            for p in pipeline.pipeline:
                name = _get_pipeline_name(p)
                if name not in self.processor_pool:
                    assert isinstance(p, DataProcessor)
                    # initialize processor and data
                    self.processor_pool[name] = p
                    self.processor_data['test'][name] = {
                        'number': 0,
                        'data': None
                    }
                    for i in range(self.num_class):
                        self.processor_data['train'][i][name] = {
                            'number': 0,
                            'data': None
                        }
                        self.processor_data['valid'][i][name] = {
                            'number': 0,
                            'data': None
                        }
    
    def add_processor(self, processor):
        assert isinstance(processor, DataProcessor)
        name = _get_pipeline_name(processor)
        self.processor_pool[name] = processor
        self.processor_data['test'][name] = {
            'number': 0,
            'data': None
        }
        for i in range(self.num_class):
            self.processor_data['train'][i][name] = {
                'number': 0,
                'data': None
            }
            self.processor_data['valid'][i][name] = {
                'number': 0,
                'data': None
            }

                
    def reset(self):
        self.train_idx = [0] * self.num_class

    def get_batch(self, pipe, batch_size):
        batch_per_class = batch_size // self.num_class
        all_data = []
        label = []
        for i in range(self.num_class):
            c, labels = self.get_batch_from_class(pipe, i, batch_per_class)
            #print(i, [len(x) if isinstance(x, list) else x.shape[0] for x in c])
            all_data.append(c)
            label.extend(labels)
        #print('before concat: len(all_data)=%d, shape=%s' % (len(all_data), str([len(x[0]) if isinstance(x[0], list) else x[0].shape[0] for x in all_data])))
        res = concat_result(all_data)
        #print('after concat: shape=%s' % (str([len(x) if isinstance(x, list) else x.shape[0] for x in res])))
        res.append(label)
        #print([len(x) if isinstance(x, list) else x.shape[0] for x in res])
        #print('[data manager] get batch: batch_size: %d, time cost %f s' % (batch_size, time2 - time1))
        return res

    def get_batch_from_class(self, pipe, class_id, class_batch):
        cur_idx = self.train_idx[class_id]
        if self.dataset_exhausted and cur_idx > len(self.train_data[class_id]):
            cur_idx = 0
        max_idx = cur_idx + class_batch
        while not self.dataset_exhausted and max_idx > len(self.train_data[class_id]):
            #self.process_dataset('train', number=max_idx - len(self.train_data[class_id]))
            self.process_dataset('train', number=max([self.extend_number, max_idx - len(self.train_data[class_id])]))
        max_idx_process = min([len(self.train_data[class_id]), max_idx])
        pipeline = self.pipeline_pool[pipe]
        idx = len(pipeline) - 1
        while idx >= 0:
            name = pipeline[idx]
            if self.processor_data['train'][class_id][name]['number'] < max_idx_process:
                idx -= 1
            else:
                break
        
        if idx == -1:
            last_value = [self.train_data[class_id]]
        else:
            name = pipeline[idx]
            last_value = self.processor_data['train'][class_id][name]['data']

        idx += 1
        while idx < len(pipeline):
            name = pipeline[idx]
            res = self.processor_pool[name].process(last_value, self.processor_data['train'][class_id][name]['number'], max_idx_process)
            self.processor_data['train'][class_id][name]['number'] = max_idx_process
            self.processor_data['train'][class_id][name]['data'] = concat_result([self.processor_data['train'][class_id][name]['data'], res])
            last_value = self.processor_data['train'][class_id][name]['data']
            idx += 1
        
        if max_idx_process == max_idx:
            # the data is enough
            res = [d[cur_idx: max_idx] for d in self.processor_data['train'][class_id][pipeline[-1]]['data']]
            self.train_idx[class_id] = max_idx
            return res, self.train_label[class_id][cur_idx : max_idx]

        idx_array = list(range(cur_idx, max_idx_process)) + (max_idx // max_idx_process - 1) * list(range(max_idx_process)) + list(range(max_idx % max_idx_process))
        self.train_idx[class_id] = max_idx % max_idx_process
        cur_class = self.train_label[class_id]
        return [slices(d, idx_array) for d in self.processor_data['train'][class_id][pipeline[-1]]['data']], cur_class[cur_idx:max_idx_process] + cur_class * (max_idx // max_idx_process - 1) + cur_class[:max_idx % max_idx_process]

    def get_batch_from_class_valid(self, pipe, class_id):
        while not self.dataset_exhausted and self.valid_batch > len(self.valid_data[class_id]):
            #self.process_dataset('valid', number = self.valid_batch - len(self.valid_data[class_id]))
            self.process_dataset('valid', number = max([self.extend_number, self.valid_batch - len(self.valid_data[class_id])]))
        if self.dataset_exhausted:
            max_valid_per_sample = int(min(self.num_count) * 0.3)
            if self.valid_batch > max_valid_per_sample:
                self.valid_batch = max_valid_per_sample
                print('[data manager] ADJUSTING VALID NUM BECAUSE DATA IS TOO SMALL %d' % (self.valid_batch))
                # we need to give back the training data
                for i in range(class_id):
                    if len(self.valid_data[i]) > self.valid_batch:
                        self.train_data[i] += self.valid_data[i][:-self.valid_batch]
                        self.train_label[i] += self.valid_label[i][:-self.valid_batch]
                        self.valid_data[i] = self.valid_data[i][-self.valid_batch:]
                        self.valid_label[i] = self.valid_label[i][-self.valid_batch:]
            if self.valid_batch > len(self.valid_data[class_id]):
                diff = self.valid_batch - len(self.valid_data[class_id])
                self.valid_data[class_id] += self.train_data[class_id][-diff:]
                self.valid_label[class_id] += self.train_label[class_id][-diff:]
                self.train_data[class_id] = self.train_data[class_id][:-diff]
                self.train_label[class_id] = self.train_label[class_id][:-diff]
        pipeline = self.pipeline_pool[pipe]
        idx = len(pipeline) - 1
        while idx >= 0:
            name = pipeline[idx]
            if self.processor_data['valid'][class_id][name]['number'] < self.valid_batch:
                idx -= 1
            else:
                break
        
        if idx == -1:
            last_value = [self.valid_data[class_id]]
        else:
            name = pipeline[idx]
            last_value = self.processor_data['valid'][class_id][name]['data']

        idx += 1
        while idx < len(pipeline):
            name = pipeline[idx]
            res = self.processor_pool[name].process(last_value, self.processor_data['valid'][class_id][name]['number'], self.valid_batch)
            self.processor_data['valid'][class_id][name]['number'] = self.valid_batch
            self.processor_data['valid'][class_id][name]['data'] = concat_result([self.processor_data['valid'][class_id][name]['data'], res])
            last_value = self.processor_data['valid'][class_id][name]['data']
            idx += 1
        
        return self.processor_data['valid'][class_id][pipeline[-1]]['data']


    def get_dataset(self, pipe, dataset='valid'):
        if dataset == 'valid':
            all_data = []
            for i in range(self.num_class):
                all_data.append(self.get_batch_from_class_valid(pipe, i))
            self.update_valid_label()
            return concat_result(all_data)

        pipeline = self.pipeline_pool[pipe]
        if self.processor_data['test'][pipeline[-1]]['number'] == 0:
            idx = len(pipeline) - 1
            while idx >= 0:
                name = pipeline[idx]
                if self.processor_data[dataset][name]['number'] == 0:
                    idx -= 1
                else:
                    break
            
            if idx == -1:
                # means the root node is not complete
                if len(self.test_data) == 0:
                    self.process_dataset('test', number=-1)
                last_value = [self.test_data]
            else:
                name = pipeline[idx]
                last_value = self.processor_data['test'][name]['data']

            idx += 1
            while idx < len(pipeline):
                name = pipeline[idx]
                last_value = self.processor_pool[name].process(last_value, 0, -1)
                self.processor_data['test'][name]['number'] = len(self.test_data)
                self.processor_data['test'][name]['data'] = last_value
                idx += 1
            
        return self.processor_data['test'][pipeline[-1]]['data']

    def get_rounded_batch_size(self, raw_batch_size, mode='unb'):
        return int(raw_batch_size // self.num_class * self.num_class)
    
    def process_dataset(self, dataset='train', number=-1):
        i = 0
        while i < number or number < 0:
            try:
                d, l = self.sess.run(self.train_next if dataset in ['train','valid'] else self.test_next)
                if dataset == 'test':
                    self.test_data.append(d[:,0,0,0].astype(np.long))
                elif dataset == 'valid':
                    for idx in range(self.num_class):
                        label = l[idx]
                        if label == 1:
                            if len(self.valid_data[idx]) >= self.valid_batch:
                                self.train_data[idx].append(d[:,0,0,0].astype(np.long))
                                self.train_label[idx].append(l)
                            else:
                                self.valid_data[idx].append(d[:,0,0,0].astype(np.long))
                                self.valid_label[idx].append(l)
                else:
                    for idx in range(self.num_class):
                        label = l[idx]
                        if label == 1:
                            self.train_data[idx].append(d[:,0,0,0].astype(np.long))
                            self.train_label[idx].append(l)
                    if sum(l) != 1:
                        self.multi_label_mode = True
                        print('[DATA MANAGER] MULTILABEL DETECTED!')
                i += 1
            except tf.errors.OutOfRangeError:
                if dataset == 'train' or dataset == 'valid':
                    self.num_count = [len(x) + len(y) for x, y in zip(self.train_data, self.valid_data)]
                    print('[data manager] DATASET EXAUSTED, distribution:', str(self.num_count))
                    self.dataset_exhausted = True
                    self.total_num = sum([len(x) for x in self.train_data])
                elif dataset == 'test':
                    self.test_count = len(self.test_data)
                break
    
    def get_meta_info(self):
        if len(self.train_data[0]) == 0:
            self.process_dataset('train', 1000)
        self.sampled_length = []
        for x in self.train_data:
            self.sampled_length.extend([len(y) for y in x[:500]])
        self.max_length = max(self.sampled_length)
        print('[data manager] original max_length: %d' % (self.max_length))
        self.avg_length = np.mean(self.sampled_length)

        
def _get_pipeline_name(processor):
    if isinstance(processor, DataProcessor):
        return processor.name
    elif isinstance(processor, str):
        return processor
    raise TypeError("get invalid processor type: ", type(processor))

def slices(datas, index):
    if isinstance(datas, list):
        return np.array(datas)[index]
    return datas[index]

class DataProcessor:
    """
    Used for processing dataset
    """
    def __init__(self, name):
        super().__init__()
        self.name = name

    def process(self, input, start_idx=0, end_idx=-1):
        raise NotImplementedError

class DataPipeline:
    """
    The pipeline of all processors
    Input should be tf.Dataset
    """
    def __init__(self, name, pipeline=[]):
        super().__init__()
        self.name = name
        self.pipeline = pipeline

    def add_processor(self, processor, index=-1):
        self.pipeline.insert(index, processor)
    
    def append_processor(self, processor):
        self.pipeline.append(processor)

class IdxToStr(DataProcessor):
    def  __init__(self, name, metadata, is_ZH):
        super().__init__(name)
        self.metadata = metadata
        vocabulary = self.metadata.get_channel_to_index_map()
        self.index_to_token = [None] * len(vocabulary)
        for token in vocabulary:
            index = vocabulary[token]
            self.index_to_token[index] = token        
        self.is_ZH = is_ZH
            
    def process(self, input, start_idx=0, end_idx=-1):
        # the input should be: [list: idxs, list: label]
        #time1 = time.time()
        strs = []
        if end_idx == -1:
            end_idx = len(input[0])
        for idx in range(start_idx, end_idx):
            if self.is_ZH:
                strs.append(''.join([self.index_to_token[int(i)] for i in input[0][idx]]))
            else:
                strs.append(' '.join([self.index_to_token[int(i)] for i in input[0][idx]]))
        #time2 = time.time()
        if self.is_ZH:
            strs = clean_zh_text(strs)
        else:
            strs = clean_en_text(strs)
        #time3 = time.time()
        #print('[idx2str] count: %d, convert time cost: %.2f s' % (len(strs), time2 - time1))
        #print('[idx2str] clean time cost: %.2f s' % (time3 - time2))
        return [strs]

class StrToSVMVec(DataProcessor):
    def __init__(self, name):
        super().__init__(name)
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process(self, input, start_idx=0, end_idx=-1):
        #time1 = time.time()
        if end_idx == -1:
            vec, self.tokenizer = vectorize_data(input[0][start_idx :], self.tokenizer)
        else:
            vec, self.tokenizer = vectorize_data(input[0][start_idx : end_idx], self.tokenizer)
        #time2 = time.time()
        #print('[str2svm] count: %d, time cost: %.2f s' % (vec.shape[0], time2 - time1))
        return [vec]

class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """
    A single set of features of data.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=128,
    output_mode="classification",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    features = []
    for (ex_index, example) in enumerate(examples):
        #if ex_index % 2000 == 0:
        #    print("Writing example %d/%d" % (ex_index, len(examples)))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = example.label
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5 and False:
            print("*** Example ***")
            print("guid: %s" % (example.guid))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            print("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    return features

class StrToBertIdx(DataProcessor):
    def __init__(self, name, tokenizer, max_length=128):
        super().__init__(name)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def process(self, input, start_idx=0, end_idx=-1):
        #time1 = time.time()
        if end_idx == -1:
            end_idx = len(input[0])
        inputExampleList = list(map(lambda x:InputExample(guid=x, text_a=input[0][x], label=0), range(start_idx, end_idx)))
        features = convert_examples_to_features(
            inputExampleList,
            self.tokenizer,
            max_length=self.max_length,
        )
        #time2 = time.time()
        #print('[str2bert] %.2f s' % (time2 - time1))
        return [features]

def to_numpy(dataset, shuffle=False):
    time1 = time.time()
    if shuffle:
        dataset = dataset.shuffle(10000, seed=1)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    X = []
    Y = []
    idx_pool = None
    count = 0
    length_pool = []
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        while True:
            try:
                example, labels = sess.run(next_element)
                X.append(example[:,0,0,0].astype(np.long))
                length_pool.append(len(X[-1]))
                Y.append(labels)
                if idx_pool is None:
                    idx_pool = [list() for _ in range(len(labels))]
                idx_pool[np.argmax(labels)].append(count)
                count += 1
            except tf.errors.OutOfRangeError:
                break
    time2 = time.time()
    print('[to numpy] count: %d, time: %.2f s' % (len(Y), time2 - time1))
    return np.array(X), np.array(Y), idx_pool, length_pool