import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import tensorflow as tf
import zipfile
import copy
import json
from transformers import (
    BertModel, 
    BertForSequenceClassification, 
    BertTokenizer, 
    BertForPreTraining, 
    BertPreTrainedModel, 
    BertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    )
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import random
import numpy as np
from nlp_utils.data_manager import DataPipeline, IdxToStr, StrToBertIdx
from nlp_utils.data_processors import *

BERT_V_ZOO = ['bert-given', 'tiny-bert-small', 'tiny-bert']
BERT_V = 1

def extract_bert_model():
    print('[BERT VERSION] ' + BERT_V_ZOO[BERT_V])
    if BERT_V == 0:
        bert_path = '/app/embedding/'
        if not os.path.exists(bert_path):
            bert_path = '/DATA-NFS/guancy/autoDL/app/embedding/'
        zipName = 'wwm_uncased_L-24_H-1024_A-16'
    elif BERT_V == 1:
        old_path = os.path.dirname(os.path.realpath(__file__))
        bert_path = '/'.join(old_path.split('/')[:-2] + ['models']) + '/'
        zipName = '2nd_General_TinyBERT_4L_312D'
    elif BERT_V == 2:
        old_path = os.path.dirname(os.path.realpath(__file__))
        bert_path = '/'.join(old_path.split('/')[:-2] + ['models']) + '/'
        zipName = '2nd_General_TinyBERT_6L_768D'
    if os.path.exists(bert_path + zipName):
        print('[extract file] will use existing files instead ...')
        return bert_path + zipName
    bert_zip = bert_path + zipName + '.zip'
    zip_file = zipfile.ZipFile(bert_zip)
    zip_file.extractall(bert_path)
    bert_path = bert_path + zipName
    if BERT_V == 0:
        os.system("mv %s/bert_config.json %s/config.json" % (bert_path, bert_path))
        os.system("mv %s/bert_model.ckpt.data-00000-of-00001 %s/model.ckpt.data-00000-of-00001" % (bert_path, bert_path))
        os.system("mv %s/bert_model.ckpt.index %s/model.ckpt.index" % (bert_path, bert_path))
        os.system("mv %s/bert_model.ckpt.meta %s/model.ckpt.meta" % (bert_path, bert_path))
    return bert_path

class BertClassification(BertPreTrainedModel):
    def __init__(self, bertModel, config, num_class):
        super(BertClassification, self).__init__(config)
        self.bert = bertModel
        self.num_labels = num_class
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_class)
        self.init_weights()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        only_last_layer=False
    ):

        if only_last_layer:
            with torch.no_grad():
                outputs = self.bert(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                )
                pooled_output = outputs[1]
                pooled_output = self.dropout(pooled_output)
        else:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

class bertModel:
    def __init__(self, metadata, timer, is_ZH, data_manager):
        super().__init__()
        self.timer = timer
        self.timer("bert-init")
        self.batch_per_train = 50
        self.batch_size_eval = 64
        self.max_seq_len = 301
        self.batch_size = 48
        self.weight_decay = 0
        self.learning_rate = 5e-5
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.
        self.total_epoch = 100
        self.logging_step = -1
        self.warmup_steps = 0
        self.metadata = metadata
        self.num_class = self.metadata.get_output_size()

        self.bert_folder = extract_bert_model()

        bertConfig = BertConfig.from_json_file(self.bert_folder + '/config.json')
        self.model = BertClassification(None, bertConfig, self.num_class)

        self.bertTokenizer = BertTokenizer.from_pretrained(self.bert_folder)
        bertModel = BertForPreTraining.from_pretrained(self.bert_folder, num_labels=self.num_class, from_tf= BERT_V == 0)
        self.model.bert = bertModel.bert
        del bertModel.cls
        self.model.to(torch.device("cuda"))
        self.data = data_manager
        self.data.add_pipeline(BertPipeline(is_ZH, metadata, self.bertTokenizer, max_length=self.max_seq_len))
        self.train_data_loader = None
        self.test_data_loader = None
        self.valid_data_loader = None
        self.done_training = False
        self.estimate_time_per_batch = None
        self.estimate_valid_time = None
        self.estimate_test_time = None
        

        # init optimizer and scheduler
        no_decay = ["bias", "LayerNorm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=self.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_epoch * self.batch_per_train
        )

        # first, we only train the classifier
        self.optimizer_only_classifier = optim.Adam(self.model.classifier.parameters(), 0.0005)

        self.place = 'cpu'

        self.timer("bert-init")
        print('[bert init] time cost: %.2f' % (self.timer.accumulation["bert-init"]))
    
    def to_deivce(self, device):
        if device == 'cuda':
            self.place = 'cuda'
            self.model.cuda()
        else:
            self.place = 'cpu'
            self.model.cpu()


    def train(self, index, epoch_key=-1):
        # feed data sequentially
        self.timer("bert-train-%d" % (epoch_key))
        print("[bert] train epoch %d begin ..." % (epoch_key))
        tr_loss, logging_loss = 0.0, 0.0
        self.model.zero_grad()
        self.model.train()
        for e in range(int(self.batch_per_train)):
            result, label = self.data.get_batch("bert", index)
            #result = self.data.process_by_index("bert", index[e * self.batch_size: (e + 1) * self.batch_size])[0]
            label = np.argmax(label, axis=1)
            device = torch.device("cuda")
            all_input_ids = torch.tensor([f.input_ids for f in result], dtype=torch.long).to(device)
            all_attention_mask = torch.tensor([f.attention_mask for f in result], dtype=torch.long).to(device)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in result], dtype=torch.long).to(device)
            all_labels = torch.tensor(label, dtype=torch.long).to(device)
            inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}
            inputs["token_type_ids"] = (all_token_type_ids)
            outputs = self.model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            
            loss.backward()

            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()

            if self.logging_step > 0 and e % self.logging_step == 0:
                logs = {}
                
                loss_scalar = (tr_loss - logging_loss) / self.logging_step
                learning_rate_scalar = self.scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                logging_loss = tr_loss

                print(json.dumps({**logs, **{"step": e}}))
                    
        self.timer("bert-train-%d" % (epoch_key))
        self.estimate_time_per_batch = self.timer.accumulation["bert-train-%d" % (epoch_key)] / self.batch_per_train
        print("[bert] epoch %d train end ... train time cost: %.2f s. time per batch %.4f s" 
            % (epoch_key, self.timer.accumulation["bert-train-%d" % (epoch_key)], self.estimate_time_per_batch))
        
    def valid(self):
        if self.valid_data_loader is None:
            # extract train dataset first
            self.timer("bert-convert-valid")
            valid_data = self.data.get_dataset("bert", "valid")[0]
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in valid_data], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in valid_data], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in valid_data], dtype=torch.long)
            valid_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
            valid_sampler = SequentialSampler(valid_data)
            self.valid_data_loader = DataLoader(valid_data, sampler=valid_sampler, batch_size=self.batch_size_eval)
            self.timer("bert-convert-valid")
            print("[bert] convert valid data done. Time cost: %.2f s" % (self.timer.get_latest_delta('bert-convert-valid')))

        self.timer("valid-bert")
        preds = None
        for idx, batch in enumerate(self.valid_data_loader):
            self.model.eval()
            
            batch = tuple(t.to(torch.device("cuda")) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                inputs["token_type_ids"] = (batch[2])
                outputs = self.model(**inputs)
                logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            
            #if idx % 100 == 0:
            #    print('[evaluate valid] %5d / %5d' % (idx, len(self.valid_data_loader)))
        exps = np.e ** preds
        delta = self.timer("valid-bert")
        print('[bert] valid time cost: %.2f s' % (delta))
        self.estimate_valid_time = delta
        return exps / np.sum(exps, axis=1, keepdims=True)

    def test(self):
        if self.test_data_loader is None:
            # extract train dataset first
            self.timer("bert-convert-test")
            test_data = self.data.get_dataset("bert", "test")[0]
            # Convert to Tensors and build dataset
            all_input_ids = torch.tensor([f.input_ids for f in test_data], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in test_data], dtype=torch.long)
            all_token_type_ids = torch.tensor([f.token_type_ids for f in test_data], dtype=torch.long)
            test_data = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
            test_sampler = SequentialSampler(test_data)
            self.test_data_loader = DataLoader(test_data, sampler=test_sampler, batch_size=self.batch_size_eval)
            self.timer("bert-convert-test")
            print("[bert] convert test data done. Time cost: %.2f s" % (self.timer.get_latest_delta('bert-convert-test')))

        self.timer("test-bert")
        preds = None
        for idx, batch in enumerate(self.test_data_loader):
            self.model.eval()
            
            batch = tuple(t.to(torch.device("cuda")) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
                inputs["token_type_ids"] = (batch[2])
                outputs = self.model(**inputs)
                logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            
            #if idx % 100 == 0:
            #    print('[evaluate] %5d / %5d' % (idx, len(self.test_data_loader)))
        exps = np.e ** preds
        delta = self.timer("test-bert")
        self.estimate_test_time = delta
        print('[bert] test time cost: %.2f s' % (delta))
        return exps / np.sum(exps, axis=1, keepdims=True)

def convert_STR_to_tensor(data, label, tokenizer, label_list, max_length=128):
    features = convert_examples_to_features(
        [InputExample(guid=i, text_a=text, label=0 if label is None else label[i]) for i, text in enumerate(data)], 
        tokenizer, 
        max_length=max_length, 
        label_list=label_list, 
        output_mode="classification")
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    
    return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

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
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    label_map = {label: i for i, label in enumerate(label_list)}

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
            label = label_map[example.label]
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

class BertPipeline(DataPipeline):
    def __init__(self, is_ZH, metadata, tokenizer, max_length):
        super().__init__("bert", pipeline=[
            ClipIdx(name='clip', number=max_length),
            IdxToStr("idx2str_bert", metadata, is_ZH),
            StrToBertIdx("str2bert", tokenizer, max_length)
        ])