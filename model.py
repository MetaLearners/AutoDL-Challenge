import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, metadata):
        """
        :param metadata: an AutoDLMetadata object. Its definition can be found in
            https://github.com/zhengying-liu/autodl_starting_kit_stable/blob/master/AutoDL_ingestion_program/dataset.py#L41
        """
        self.domain = infer_domain(metadata)
        print("[task selection] the current domain is: " + self.domain)
        if self.domain == 'image':
            from model_image import CVModel
            self.model = CVModel(metadata)
        elif self.domain == 'video':
            from model_video import CVModel
            self.model = CVModel(metadata)
        elif self.domain == 'tabular':
            from model_tabular import tabularModel
            self.model = tabularModel(metadata)
        elif self.domain == 'text':
            from model_nlp_bert import nlpModel
            self.model = nlpModel(metadata)
        elif self.domain == 'speech':
            from model_speech import speechModel
            self.model = speechModel(metadata)
        else:
            raise ValueError("[task selection] no fit model found for domain: " + self.domain)
        self.done_training = False
        # the loop of calling 'train' and 'test' will only run if self.done_training = False
        # otherwinse, the looop will go until the time budge in used up set self.done_training = True
        # when you think the model is converged or when is not enough time for next round of traning

    def train(self, dataset, remaining_time_budget=None):
        """
        :param dataset: a `tf.data.Dataset` object. Each of its examples is of the form (example, labels)
            where `example` is a dense 4-D Tensor of shape  (sequence_size, row_count, col_count, num_channels)
            and `labels` is a 1-D Tensor of shape (output_dim,)
            Here `output_dim` represents number of classes of this multilabel classification task.
        :param remaining_time_budget: a float, time remaining to execute train()
        :return: None
        """
        self.model.train(dataset, remaining_time_budget)
        self.done_training = self.model.done_training

    def test(self, dataset, remaining_time_budget=None):
        """
        :param: Same as that of `train` method, except that the labes will be empty (all zeros)
        :return: predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim)
            The values should be binary or in the interval [0,1].
        """
        result = self.model.test(dataset, remaining_time_budget)
        self.done_training = self.model.done_training
        return result

def infer_domain(metadata):
    """Infer the domain from the shape of the 4-D tensor."""
    row_count, col_count = metadata.get_matrix_size(0)
    sequence_size = metadata.get_sequence_size()
    channel_to_index_map = dict(metadata.get_channel_to_index_map())
    domain = None
    if sequence_size == 1:
      if row_count == 1 or col_count == 1:
        domain = "tabular"
      else:
        domain = "image"
    else:
      if row_count == 1 and col_count == 1:
        if channel_to_index_map:
          domain = "text"
        else:
          domain = "speech"
      else:
        domain = "video"
    return domain

