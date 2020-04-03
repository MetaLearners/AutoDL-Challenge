import numpy as np
import scipy
from scipy.sparse import vstack, hstack
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from skeleton.utils.timer import Timer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import roc_auc_score
from nlp_utils.data_manager import DataPipeline, IdxToStr, StrToSVMVec, DataProcessor
from nlp_utils.data_processors import SIdx2SVMVec, Idx2SIdx

class svmModel:
    def __init__(self, timer, metadata, is_ZH, data_manager, zoo):
        super().__init__()
        self.timer = timer
        self.timer("svm init")
        model = LinearSVC(random_state=0, tol=1e-5, max_iter=500)
        self.model = CalibratedClassifierCV(model)
        self.model_multi_class = RandomForestClassifier(n_jobs=-1)
        delta = self.timer("svm init")
        self.data = data_manager
        self.zoo = zoo
        svmPipe = DataPipeline('svm', pipeline=[
            Idx2SIdx(None, 'idx2sidxLower'),
            SIdx2SVMVec(),
        ])
        self.data.add_pipeline(svmPipe)
        print('[svm] init model. time cost: %.2f s' % (delta))

    def train(self, index, epoch_key=-1):
        self.timer("svm fit %d" % (epoch_key))

        self.timer("svm fit data load %d" % (epoch_key))
        print("[svm] load data begin.")
        data_lower = self.data.get_batch("idx2idxLowerPipe", index)
        labels = data_lower[1]
        # generate tokenizer
        self.data.processor_pool['sidx2svm'].max_dim = len(self.zoo.new_vocab_idx_map)
        vec = self.data.processor_pool['sidx2svm'].fit_transform(data_lower[0])
        self.data.reset()
        delta = self.timer("svm fit data load %d" % (epoch_key))
        print("[svm] load data. time cost: %.2f s" % (delta))

        self.timer("svm train %d" % (epoch_key))
        if not self.data.multi_label_mode:
            self.model.fit(vec, np.argmax(labels, axis=1))
        else:
            self.model_multi_class.fit(vec, labels)
        delta = self.timer("svm train %d" % (epoch_key))
        print("[svm] train svm. time cost: %.2f s" % (delta))
        
        self.timer("svm fit %d" % (epoch_key))
        print('[svm fit] total time cost: %.2f s' % (self.timer.accumulation["svm fit %d" % (epoch_key)]))

    def valid(self):
        self.timer("svm valid")
        self.timer("svm valid dataload")
        valid_data = self.data.get_dataset("svm", "valid")
        valid_data = valid_data[0]
        delta = self.timer("svm valid dataload")
        print("[svm] transform valid data. time cost: %.2f s" % (delta))
        if self.data.multi_label_mode:
            res = self.model_multi_class.predict(valid_data)
            #res = np.concatenate([x[:,0:1] for x in self.model_multi_class.predict_proba(valid_data)], axis=1)
        else:
            res = self.model.predict_proba(valid_data)
        self.timer("svm valid")
        print('[svm valid] time cost: %.2f s' % (self.timer.get_latest_delta("svm valid")))
        return res

    def predict(self):
        self.timer("svm predict")
        self.timer("svm test dataload")
        test_data = self.data.get_dataset("svm", "test")
        test_data = test_data[0]
        delta = self.timer("svm test dataload")
        print("[svm] transform test data. time cost: %.2f s" % (delta))
        if self.data.multi_label_mode:
            res = self.model_multi_class.predict(test_data)
            #res = np.concatenate([x[:,0:1] for x in self.model_multi_class.predict_proba(test_data)], axis=1)
        else:
            res = self.model.predict_proba(test_data)
        self.timer("svm predict")
        print("[svm predict] time cost: %.2f s" % (self.timer.get_latest_delta("svm predict")))
        return res

class SVMPipeline(DataPipeline):
    def __init__(self, metadata, is_ZH):
        super().__init__("svm", pipeline=[
            IdxToStr("idx2str", metadata, is_ZH),
            StrToSVMVec("str2svm")
        ])

# there is a bug when building StrToSVMVec's tokenizer
# we use another pipeline to fix it
class SVMFixPipeline(DataPipeline):
    def __init__(self):
        super().__init__("svmfix", pipeline=["idx2str"])