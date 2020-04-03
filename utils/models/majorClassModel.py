import tensorflow as tf
import numpy as np

class majorClassModel:
    def __init__(self, metadata):
        self.metadata = metadata
        self.num_class = metadata.get_output_size()
        self.majorClass = None
    
    def train(self, dataset, remaining_time_budget=None):
        try:
            count = dataset.reduce(np.zeros(self.num_class).astype(np.float32), lambda state, x: state + x[1])
        except:
            count = dataset.reduce(np.zeros(self.num_class), lambda state, x: state + x[1])
        with tf.Session() as sess:
            c = sess.run(count)
            self.majorClass = np.argmax(c)
            print('[train meta] total example: %d' % (sum(c)))
            print('[train meta] class distribution:', c.tolist())
        
    def test(self, dataset, remaining_time_budget=None):
        count_test_num = dataset.reduce(0, lambda state, x: state + 1)
        with tf.Session() as sess:
            num = sess.run(count_test_num)
            print('[test  meta] total example: %d' % (num))
        result = np.zeros((num, self.num_class))
        result[:,self.majorClass] = 1
        return result