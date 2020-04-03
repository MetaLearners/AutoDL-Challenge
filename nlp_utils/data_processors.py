import array
from .data_manager import DataPipeline, DataProcessor
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
from keras.preprocessing import text
from keras.preprocessing import sequence

class Idx2SIdx(DataProcessor):
    def __init__(self, re_index_list, name='idx2sidx'):
        super().__init__(name)
        self.re_index = re_index_list

    def clean_seq(self, seq):
        mapped = self.re_index[seq]
        return mapped[mapped != -1]
        new_seq = []
        for e in seq:
            new_i = self.re_index[e]
            if new_i != -1:
                new_seq.append(new_i)
        return new_seq

    def process(self, input, start_idx=0, end_idx=-1):
        if end_idx == -1:
            end_idx = len(input[0])
        returned_line = [None] * (end_idx - start_idx)
        for idx in range(start_idx, end_idx):
            returned_line[idx - start_idx] = self.clean_seq(input[0][idx])#self.re_index[input[0][idx]]
        return [returned_line]

def padding(ele, max_length=128, padIdx=0):
    if len(ele) >= max_length:
        return ele[:max_length]
    if isinstance(ele, list):
        ele += [padIdx] * (max_length - len(ele))
        return ele
    else:
        ele = ele.tolist()
        ele += [padIdx] * (max_length - len(ele))
        return ele
    
class IdxPadding(DataProcessor):
    def __init__(self, name='idxpadding', max_length=0, padIdx=-1):
        super().__init__(name)
        self.max_length = max_length
        self.padIdx = padIdx
    
    def process(self, input, start_idx=0, end_idx=-1):
        if end_idx < 0:
            end_idx = len(input[0])
        lengths = [min([len(a), self.max_length]) for a in input[0][start_idx : end_idx]]
        test_data = [padding(a, self.max_length, self.padIdx) for a in input[0][start_idx : end_idx]]
        return [test_data, lengths]

def _make_int_array():
    return array.array(str("i"))

class SIdx2SVMVec(DataProcessor):
    def __init__(self, name='sidx2svm'):
        super().__init__(name)
        self.transformer = TfidfTransformer()
        self.max_dim = 0
    
    def set_transformer(self, tran):
        self.transformer = tran
    
    def _count_vectorize(self, datas):
        assert self.max_dim > 0
        j_indices = []
        indptr = []
        values = _make_int_array()
        indptr.append(0)
        for data in datas:
            feature_counter = {}
            for idx in data:
                if idx in feature_counter:
                    feature_counter[idx] += 1
                else:
                    feature_counter[idx] = 1
            j_indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(j_indices))
        
        if indptr[-1] > np.iinfo(np.int32).max:  # = 2**31 - 1
            indices_dtype = np.int64
        else:
            indices_dtype = np.int32
        j_indices = np.asarray(j_indices, dtype=indices_dtype)
        indptr = np.asarray(indptr, dtype=indices_dtype)
        values = np.frombuffer(values, dtype=np.intc)

        X = csr_matrix((values, j_indices, indptr),
                            shape=(len(indptr) - 1, self.max_dim),
                            dtype=np.int64)
        X.sort_indices()
        return X

    def fit_transform(self, datas):
        tmp = self._count_vectorize(datas)
        return self.transformer.fit_transform(tmp)

    def process(self, input, start_idx=0, end_idx=-1):
        if end_idx == -1:
            end_idx = len(input[0])
        return [self.transformer.transform(self._count_vectorize(input[0][start_idx: end_idx]))]

class strToKerasTokenizer(DataProcessor):
    def __init__(self, name='strKerasTokenizer'):
        super().__init__(name)
        self.tokenizer = None
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def process(self, input, start_idx=0, end_idx=-1):
        if end_idx == -1:
            end_idx = len(input[0])
        x_train = self.tokenizer.texts_to_sequences(input[0][start_idx:end_idx])
        return [x_train]

class kerasPadding(DataProcessor):
    def __init__(self, name='kerasPadding'):
        super().__init__(name)
        self.max_length = 0
        
    def process(self, input, start_idx=0, end_idx=-1):
        if end_idx == -1:
            end_idx = len(input[0])
        
        x_train = sequence.pad_sequences(input[0][start_idx:end_idx], maxlen=self.max_length)

        return [x_train] # this is an ndarray

class ClipIdx(DataProcessor):
    def __init__(self, name='clip301', number=301):
        super().__init__(name)
        self.number = 301
    
    def process(self, input, start_idx=0, end_idx=-1):
        end_idx = end_idx if end_idx >= 0 else len(input[0])
        return [[x[:self.number] for x in input[0][start_idx:end_idx]]]

class ToNumpy(DataProcessor):
    def __init__(self):
        super().__init__('tonumpy')
    
    def process(self, input, start_idx=0, end_idx=-1):
        end_idx = end_idx if end_idx >= 0 else len(input[0])
        return [np.array(input[start_idx:end_idx], dtype=object)]
