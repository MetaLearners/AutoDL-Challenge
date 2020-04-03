import keras
from keras.layers import Input, LSTM, Dense
from keras.layers import Dropout, Masking
from keras.layers import Embedding, Flatten, Conv1D, concatenate
from keras.layers import SeparableConv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D, GlobalMaxPool1D, SpatialDropout1D, Activation, Add, GlobalMaxPooling1D, add, BatchNormalization
import numpy as np
from nlp_utils.data_manager import DataProcessor, DataPipeline, nlp_dataset, IdxToStr
from nlp_utils.data_processors import *
from keras import regularizers
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow as tf
import gc
import time
from nlp_utils.CONSTANT import *
from nlp_utils.scheduler import lr_scheduler
    
class kerasCnnModel:
    def __init__(self, metadata, data, fasttext_embeddings_index, num_class, sess, is_ZH):
        time1 = time.time()
        self.data = data
        self.metadata = metadata
        self.is_ZH = is_ZH
        self.data.add_pipeline(DataPipeline(
            'cnnFix',
            [
                IdxToStr('idx2str', metadata, is_ZH),
            ]
        ))
        self.cur_length = 0
        self.model = None
        self.word_index = None
        self.num_features = None
        self.num_class = num_class
        self.num_epoch = 1
        self.valid_ratio = 0.1
        self.batch_size = 256

        self.estimate_time_per_batch = None
        self.estimate_valid_time = None
        self.estimate_test_time = None

        self.fasttext_embeddings_index = fasttext_embeddings_index
        if self.num_class < 5:
            self.scheduler = lr_scheduler(lr=0.001, patience=0, threshold=0.01, min_lr=0.0005, rate=0.5, cooldown=0)
            self.init_lr = 0.001
        elif self.num_class < 50:
            self.scheduler = lr_scheduler(lr=0.005, patience=0, threshold=0.01, min_lr=0.001, rate=0.2, cooldown=0)
            self.init_lr = 0.005
        else:
            self.scheduler = lr_scheduler(lr=0.01, patience=0, threshold=0.01, min_lr=0.001, rate=0.5, cooldown=0)
            self.init_lr = 0.01
        self.pre_cumpute_model()
        self.build_model = False
        time2 =  time.time()
        print('[keras cnn] init model time cost %.2f s' % (time2 - time1))

    def set_multi_length(self, length_seq):
        self.length_seq = length_seq
        if self.cur_length == 0:
            for length in length_seq:
                self.data.add_pipeline(DataPipeline(
                    'kerasCnn_%d' % (length),
                    [
                        ClipIdx(name='clip_%d' % (length), number=length),
                        IdxToStr('idx2str_%d' % (length), self.metadata, self.is_ZH),
                        strToKerasTokenizer(name='tokenizer_%d' % (length)),
                        IdxPadding(name='idxpadding_%d' % (length), max_length=length)
                    ]
                ))
            self.cur_length = length_seq[0]

    def train(self, batch_size, epoch_key=-1):
        if self.build_model == False:
            # we need to build model first
            time1 = time.time()
            self.data.reset()
            tokenizer_sample_str = min([TOKENIZER_SAMPLE_STR, TOKENIZER_MAX_TOKEN // self.data.avg_length])
            tokenizer_sample_str = self.data.get_rounded_batch_size(tokenizer_sample_str)
            strs = self.data.get_batch('cnnFix', tokenizer_sample_str)[0]
            tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE, char_level=self.is_ZH, filters='[“”【】/（）：「」、|，；。"/\\(){}[]|@,;]."#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
            tokenizer.fit_on_texts(strs)
            self.word_index = tokenizer.word_index
            self.num_features = min(len(self.word_index) + 1, MAX_VOCAB_SIZE)
            for length in self.length_seq:
                self.data.processor_pool['tokenizer_%d' % (length)].tokenizer = tokenizer
            self.data.reset()
            time9 = time.time()
            print('[keras init] get meta data time cost: %.2f s' % (time9 - time1))
            self.generate_emb_matrix()
            self.rebuild_model(self.embedding_matrix)
            time10 = time.time()
            print('[keras init] build embedding matrix time cost: %.2f s' % (time10 - time9))
            self.build_model = True
            '''
            self.model = kerasCnnModel.text_cnn_model(
                embedding_matrix=self.embedding_matrix, 
                num_features=self.num_features,
                num_classes=self.num_class,
            )
            '''
            '''
            self.model.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.RMSprop(),
                metrics=["accuracy"])
            '''
            time2 = time.time()
            print('[keras cnn] build model time cost: %.2f s, total time cost: %.2f s' % (time2 - time10, time2 - time1))
        time1 = time.time()
        train_x, _, train_y = self.data.get_batch('kerasCnn_%d' % (self.cur_length), batch_size)
        train_x = np.array(train_x)
        times = time.time()
        if self.data.multi_label_mode:
            model = self.model_multi_class
        else:
            model = self.model
        model.fit(
            train_x,
            np.array(train_y),
            epochs=self.num_epoch,
            callbacks=None,
            validation_data=None,
            verbose=0,
            batch_size=self.batch_size,
            shuffle=True)
        time2 = time.time()
        print('[keras cnn] train model epoch %d time cost %.2f s - FIT TIME COST %.2f s' % (epoch_key, time2 - time1, time2 - times))
        self.estimate_time_per_batch = time2 - time1


    def valid(self):
        time1 = time.time()
        data = self.data.get_dataset('kerasCnn_%d' % (self.cur_length), 'valid')[0]
        time3 = time.time()
        if self.data.multi_label_mode:
            model = self.model_multi_class
        else:
            model = self.model
        result = model.predict(np.array(data), batch_size=self.batch_size * 4)
        time2 = time.time()
        print('[keras cnn] valid time cost: %.2f s, MODEL TIME: %.2f s' % (time2 - time1, time2 - time3))
        return result

    def predict(self):
        time1 = time.time()
        data = self.data.get_dataset('kerasCnn_%d' % (self.cur_length), 'test')[0]
        if self.data.multi_label_mode:
            model = self.model_multi_class
        else:
            model = self.model
        result = model.predict(np.array(data), batch_size=self.batch_size * 4)
        print(sum(result[0]))
        time2 = time.time()
        print('[keras cnn] test time cost: %.2f s' % (time2 - time1))
        return result

    def clear_gpu(self):
        print('[keras cnn] clear gpu')
        del self.model
        del self.model_multi_class
        gc.collect()
        K.clear_session()
        print('[keras cnn] clear gpu end')

    def generate_emb_matrix(self):
        time1 = time.time()
        cnt = 0
        self.embedding_matrix = np.zeros((MAX_VOCAB_SIZE, EMBEDDING_DIM))
        for word, i in self.word_index.items():
            if i >= self.num_features:
                continue
            embedding_vector = self.fasttext_embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                self.embedding_matrix[i] = embedding_vector
            else:
                # self.embedding_matrix[i] = np.zeros(300)
                self.embedding_matrix[i] = np.random.uniform(
                    -0.05, 0.05, size=EMBEDDING_DIM)
                cnt += 1
        time2 = time.time()
        print('[keras cnn] fastText oov words: %s, generate emb time cost %.2f s' % (cnt, time2 - time1))

    def pre_cumpute_model(self):
        self.inputs = Input(name='inputs', shape=[None])
        self.embedding = Embedding(input_dim=20000, output_dim=300)
        layer = self.embedding(self.inputs)
        self.convLayers = []
        self.poolLayers = []
        filter_sizes = [2,3,4,5]
        cnns = []
        for size in filter_sizes:
            self.convLayers.append(Conv1D(128, size, padding='same', strides=1, activation='relu'))
            self.poolLayers.append(GlobalMaxPool1D())
            cnns.append(self.poolLayers[-1](self.convLayers[-1](layer)))
        
        cnn_merge = concatenate(cnns, axis=-1)
        self.drop = Dropout(0.20)
        out = self.drop(cnn_merge)
        self.dense = Dense(self.num_class)
        main_out = Activation('softmax')(self.dense(out))
        
        main_out_multi_class = Activation('sigmoid')(self.dense(out))

        self.model = keras.models.Model(inputs=self.inputs, outputs=main_out)

        self.opt = keras.optimizers.RMSprop(0.0)

        self.model.compile(loss="categorical_crossentropy",
                optimizer=self.opt,
                metrics=["accuracy"])
        time1 = time.time()
        self.model.fit(np.random.randint(20000, size=(get_proper_data(self.num_class), 500)),
            np.eye(self.num_class)[np.random.randint(self.num_class, size=(get_proper_data(self.num_class)))],
            epochs=self.num_epoch,
            callbacks=None,
            validation_data=None,
            verbose=0,
            batch_size=256,
            shuffle=True)
        time2 = time.time()
        self.model.predict(np.array(np.random.randint(15000, size=(get_valid_per_sample(self.num_class) * self.num_class, 500))), batch_size=256 * 4)
        time3 = time.time()
        print('[TESTING] %.2f s --- %.2f s' % (time2- time1, time3- time2))

        self.model_multi_class = keras.models.Model(inputs=self.inputs, outputs=main_out_multi_class)
        self.model_multi_class.compile(loss='binary_crossentropy', optimizer=self.opt, metrics=['accuracy'])
        time1 = time.time()
        self.model_multi_class.fit(np.random.randint(20000, size=(get_proper_data(self.num_class), 500)),
            np.random.randint(1, size=(get_proper_data(self.num_class), self.num_class)),
            epochs=self.num_epoch,
            callbacks=None,
            validation_data=None,
            verbose=0,
            batch_size=256,
            shuffle=True)
        time2 = time.time()
        self.model_multi_class.predict(np.array(np.random.randint(15000, size=(get_valid_per_sample(self.num_class) * self.num_class, 500))), batch_size=256 * 4)
        time3 = time.time()
        print('[TESTING] %.2f s --- %.2f s' % (time2- time1, time3- time2))

        K.set_value(self.opt.lr, self.init_lr)
    
    def rebuild_model(self, embedding_matrix):
        self.embedding.set_weights([embedding_matrix])

    @staticmethod
    def deprecated(embedding_matrix,
                       num_features,
                       num_classes,
                       input_tensor=None,
                       filters=128,
                       emb_size=300,
                       ):

        inputs = Input(name='inputs', shape=[None], tensor=input_tensor)
        if embedding_matrix is None:
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size)(inputs)
        else:
            num_features = MAX_VOCAB_SIZE
            layer = Embedding(input_dim=num_features,
                              output_dim=emb_size,
                              embeddings_initializer=keras.initializers.Constant(
                                  embedding_matrix))(inputs)

        cnns = []
        filter_sizes = [2, 3, 4, 5]
        for size in filter_sizes:
            cnn_l = Conv1D(filters,
                           size,
                           padding='same',
                           strides=1,
                           activation='relu')(layer)
            pooling_l = GlobalMaxPool1D()(cnn_l)
            cnns.append(pooling_l)
            #cnns.append(cnn_l)

        cnn_merge = concatenate(cnns, axis=-1)
        out = Dropout(0.2)(cnn_merge)
        #out = AttentionLayer()([inputs, out])
        main_output = Dense(num_classes, activation='softmax')(out)
        #print(K.int_shape(main_output))
        model = keras.models.Model(inputs=inputs, outputs=main_output)
        return model
