# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import threading
import random

import tensorflow as tf
import torch
import torchvision as tv
import numpy as np

import skeleton_video
from architectures_video.resnet import ResNet18
from skeleton_video.projects import LogicModel, get_logger
from skeleton_video.projects.others import NBAC, AUC


torch.backends.cudnn.benchmark = True
threads = [
    threading.Thread(target=lambda: torch.cuda.synchronize()),
    threading.Thread(target=lambda: tf.Session())
]
[t.start() for t in threads]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LOGGER = get_logger(__name__)


def set_random_seed_all(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_random_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CVModel(LogicModel):
    def __init__(self, metadata):
        # set_random_seed_all(0xC0FFEE)
        super(CVModel, self).__init__(metadata)
        self.use_test_time_augmentation = False
        self.update_transforms = False

    def build(self): 
        base_dir = os.path.dirname(os.path.abspath(__file__))
        in_shape = self.info['dataset']['shape']
        num_class = self.info['dataset']['num_class']
        # torch.cuda.synchronize()
        
        LOGGER.info('[init] session')
        [t.join() for t in threads]

        self.device = torch.device('cuda', 0)
        self.session = tf.Session()

        LOGGER.info('[init] Model')
        Network = ResNet18  # ResNet18  # BasicNet, SENet18, ResNet18
        self.model = Network(in_shape, num_class)
        # self.model_pred = Network(in_shape, num_class).eval()
        # torch.cuda.synchronize()

        LOGGER.info('[init] weight initialize')
        if Network in [ResNet18]:
            model_path = os.path.join(base_dir, 'models')
            LOGGER.info('model path: %s', model_path)
            self.model.init(model_dir=model_path, gain=1.0)
        else:
            self.model.init(gain=1.0)
        torch.cuda.synchronize()

        LOGGER.info('[init] copy to device')
        self.model = self.model.to(device=self.device, non_blocking=True) #.half()
        # self.model_pred = self.model_pred.to(device=self.device, non_blocking=True) #.half()
        self.is_half = self.model._half
        torch.cuda.synchronize()
        self.start_fc_epoch = 1
        self.start_conv_epoch = 1

        LOGGER.info('[init] done.')

    def update_model(self):
        num_class = self.info['dataset']['num_class']

        epsilon = min(0.1, max(0.001, 0.001 * pow(num_class / 10, 2)))
        if self.is_multiclass():
            self.model.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            self.tau = 8.0
            LOGGER.info('[update_model] %s (tau:%f, epsilon:%f)', self.model.loss_fn.__class__.__name__, self.tau, epsilon)
        else:
            self.model.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            self.tau = 8.0
            LOGGER.info('[update_model] %s (tau:%f, epsilon:%f)', self.model.loss_fn.__class__.__name__, self.tau, epsilon)
        # self.model_pred.loss_fn = self.model.loss_fn

        if self.is_video(): 
            # not use fast auto aug
            self.hyper_params['conditions']['use_fast_auto_aug'] = False
            times = self.hyper_params['dataset']['input'][0]
            self.model.set_video(times=times)
            # self.model_pred.set_video(times=times)

        self.init_opt()
        LOGGER.info('[update] done.')

    def init_opt(self):
        steps_per_epoch = self.hyper_params['dataset']['steps_per_epoch']
        batch_size = self.hyper_params['dataset']['batch_size']

        LOGGER.info('init optimizer start~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        self.params_simple = [p for n, p in self.model.named_parameters() if p.requires_grad and 'simple' == n[:6] ] # or 'conv1d' == n[:6]
        self.params_fc = [p for n, p in self.model.named_parameters() if p.requires_grad and ( 'fc' == n[:2]) ]
        self.params_conv = [p for n, p in self.model.named_parameters() if p.requires_grad and ( n[:5] =='layer' or n[:5]=='conv1' or n[:2]=='bn') ]
        # self.params_layer1

        self.params_keys = [n for n, p in self.model.named_parameters() if p.requires_grad and (n[:6] != 'simple') ]
        simple_keys = [n for n, p in self.model.named_parameters() if p.requires_grad and 'simple' == n[:6] ]
        fc_keys = [n for n, p in self.model.named_parameters() if p.requires_grad and 'fc' == n[:2] ]
        conv_keys = [n for n, p in self.model.named_parameters() 
            if p.requires_grad and (n[:5] =='layer' or n[:5]=='conv1' or n[:2]=='bn' or n[:3]=='rnn') 
        ]
        
        LOGGER.info('~~~~~~~~~~~~~~~~~model parameters~~~~~~~~~~~~~~~~~ simple:\n %s, ', simple_keys)
        LOGGER.info('~~~~~~~~~~~~~~~~~model parameters~~~~~~~~~~~~~~~~~ fc:\n %s, ', fc_keys)
        LOGGER.info('~~~~~~~~~~~~~~~~~model parameters~~~~~~~~~~~~~~~~~ conv:\n %s, ', conv_keys)
        LOGGER.info('~~~~~~~~~~~~~~~~~model parameters~~~~~~~~~~~~~~~~~ all:\n %s, ', self.params_keys)

        init_lr = self.hyper_params['optimizer']['lr']
        self.lr_table = {
            'small_image':{
                'lr_simple':0.001, 'lr_fc_only':0.001, 'lr_fc':0.01, 'lr_conv':0.002
                },
            'large_image':{
                'lr_simple':0.05, 'lr_fc_only':0.025, 'lr_fc':0.01, 'lr_conv':0.002
                },
            'very_large_image':{
                'lr_simple':0.2, 'lr_fc_only':0.01, 'lr_fc':0.01, 'lr_conv':0.002
            }
        }
        
        self.build_optimizer()

    def adapt(self, remaining_time_budget=None):
        epoch = self.info['loop']['epoch']
        input_shape = self.hyper_params['dataset']['input']
        height, width = input_shape[:2]
        batch_size = self.hyper_params['dataset']['batch_size']

        train_score = np.average([c['train']['score'] for c in self.checkpoints[-5:]])
        valid_score = np.average([c['valid']['score'] for c in self.checkpoints[-5:]])
        LOGGER.info('[adapt] [%04d/%04d] train:%.3f valid:%.3f',
                    epoch, self.hyper_params['dataset']['max_epoch'],
                    train_score, valid_score)

        self.use_test_time_augmentation = self.info['loop']['test'] > 1

        # if self.hyper_params['conditions']['use_fast_auto_aug']:
        #     self.hyper_params['conditions']['use_fast_auto_aug'] = valid_score > 0.995
        self.hyper_params['conditions']['use_fast_auto_aug'] = False
        # Adapt Apply Fast auto aug
        if self.hyper_params['conditions']['use_fast_auto_aug'] and \
                (train_score > 0.95 or self.info['terminate']) and \
                remaining_time_budget > 120 and \
                valid_score > 0.9 and \
                self.dataloaders['valid'] is not None and \
                not self.update_transforms:
            LOGGER.info('[adapt] search fast auto aug policy')
            self.update_transforms = True
            self.info['terminate'] = True

            original_valid_policy = self.dataloaders['valid'].dataset.transform.transforms
            policy = skeleton_video.data.augmentations.autoaug_policy()

            num_policy_search = 100
            num_sub_policy = 3
            num_select_policy = 5
            searched_policy = []
            for policy_search in range(num_policy_search):
                selected_idx = np.random.choice(list(range(len(policy))), num_sub_policy)
                selected_policy = [policy[i] for i in selected_idx]

                self.dataloaders['valid'].dataset.transform.transforms = original_valid_policy + [
                    lambda t: t.cpu().float() if isinstance(t, torch.Tensor) else torch.Tensor(t),
                    tv.transforms.ToPILImage(),
                    skeleton_video.data.augmentations.Augmentation(
                        selected_policy
                    ),
                    tv.transforms.ToTensor(),
                    lambda t: t.to(device=self.device) #.half()
                ]

                metrics = []
                for policy_eval in range(num_sub_policy * 2):
                    valid_dataloader = self.build_or_get_dataloader('valid', self.datasets['valid'], self.datasets['num_valids'])
                    # original_valid_batch_size = valid_dataloader.batch_sampler.batch_size
                    # valid_dataloader.batch_sampler.batch_size = batch_size

                    valid_metrics = self.epoch_valid(self.info['loop']['epoch'], valid_dataloader, reduction='max')

                    # valid_dataloader.batch_sampler.batch_size = original_valid_batch_size
                    metrics.append(valid_metrics)
                loss = np.max([m['loss'] for m in metrics])
                score = np.max([m['score'] for m in metrics])
                LOGGER.info('[adapt] [FAA] [%02d/%02d] score: %f, loss: %f, selected_policy: %s',
                            policy_search, num_policy_search, score, loss, selected_policy)

                searched_policy.append({
                    'loss': loss,
                    'score': score,
                    'policy': selected_policy
                })

            flatten = lambda l: [item for sublist in l for item in sublist]

            # filtered valid score
            searched_policy = [p for p in searched_policy if p['score'] > valid_score]

            if len(searched_policy) > 0:
                policy_sorted_index = np.argsort([p['score'] for p in searched_policy])[::-1][:num_select_policy]
                # policy_sorted_index = np.argsort([p['loss'] for p in searched_policy])[:num_select_policy]
                policy = flatten([searched_policy[idx]['policy'] for idx in policy_sorted_index])
                policy = skeleton_video.data.augmentations.remove_duplicates(policy)

                LOGGER.info('[adapt] [FAA] scores: %s', [searched_policy[idx]['score'] for idx in policy_sorted_index])

                original_train_policy = self.dataloaders['train'].dataset.transform.transforms
                self.dataloaders['train'].dataset.transform.transforms = original_train_policy + [
                    lambda t: t.cpu().float() if isinstance(t, torch.Tensor) else torch.Tensor(t),
                    tv.transforms.ToPILImage(),
                    skeleton_video.data.augmentations.Augmentation(
                        policy
                    ),
                    tv.transforms.ToTensor(),
                    lambda t: t.to(device=self.device) #.half()
                ]

            self.dataloaders['valid'].dataset.transform.transforms = original_valid_policy
            # self.dataloaders['test'].dataset.transform.transforms = original_valid_policy

            # reset optimizer pararms
            # self.model.init()
            self.hyper_params['optimizer']['lr'] /= 2.0
            # self.init_opt()
            self.hyper_params['conditions']['max_inner_loop_ratio'] *= 3
            self.hyper_params['conditions']['threshold_valid_score_diff'] = 0.00001
            self.hyper_params['conditions']['min_lr'] = 1e-8

    def activation(self, logits):
        if self.is_multiclass():
            logits = torch.sigmoid(logits)
            prediction = (logits > 0.5).to(logits.dtype)
        else:
            logits = torch.softmax(logits, dim=-1)
            _, k = logits.max(-1)
            prediction = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device).scatter_(-1, k.view(-1, 1), 1.0)
        return logits, prediction

    def build_optimizer(self):
        LOGGER.info('this data set is %s', self.image_mode)
        self.optimizer_simple = torch.optim.SGD(
            self.params_simple, lr=self.lr_table[self.image_mode]['lr_simple'], momentum=0.9, nesterov=False, weight_decay=0.01, 
            )
        self.lr_schedular_simple = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_simple, mode='max', factor=0.5, patience=3, verbose=False
            )

        self.optimizer_fc = torch.optim.SGD(
            self.params_fc, lr=self.lr_table[self.image_mode]['lr_fc_only'], momentum=0.9, nesterov=False, weight_decay=0.01, 
        )
        self.lr_schedular_fc = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_fc, mode='max', factor=0.5, patience=2, verbose=True
            )

        self.optimizer_all = torch.optim.SGD(
            [
                {'params':self.params_fc, 'lr':self.lr_table[self.image_mode]['lr_fc']},
                {"params":self.params_conv, 'lr':self.lr_table[self.image_mode]['lr_conv']}
            ],
            lr=0.01, momentum=0.9, weight_decay=0.01, nesterov=True
        )
        self.lr_schedular_all = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_all, mode='max', factor=0.5, patience=4, verbose=True, min_lr=1e-8, cooldown=0, threshold=0.02
        )

    def epoch_train(self, epoch, train, model=None, optimizer=None):
        model = model if model is not None else self.model

        if epoch == 0:  # update image mode
            self.build_optimizer()
        
        if self.image_mode == 'small_image':
            self.start_fc_epoch = 0
            self.start_conv_epoch = 0
            batch_num_per_epoch = [10, 15, 20, 25]
        elif self.image_mode == ' large_image':
            self.start_fc_epoch = 0
            self.start_conv_epoch = 0
            batch_num_per_epoch = [10, 15, 20, 25]
        else:
            self.start_fc_epoch = 0
            self.start_conv_epoch = 0
            batch_num_per_epoch = [10, 15, 20, 25]

        if epoch < self.start_fc_epoch and not self.is_video():
            optimizer= optimizer if optimizer is not None else self.optimizer_simple
        elif epoch < self.start_conv_epoch and not self.is_video():
            optimizer = optimizer if optimizer is not None else self.optimizer_fc
        else:
            optimizer = optimizer if optimizer is not None else self.optimizer_all

        if epoch >= self.start_fc_epoch:
            self.model.set_use_linear_only(False)
        # if epoch >= start_fc_epoch:
        #     self.model_pred.set_use_linear_only(False)

        if epoch < len(batch_num_per_epoch):
            train.set_steps(batch_num_per_epoch[epoch])
        else:
            train.change_steps(step_num=2)
        
        model.train()
        model.zero_grad()
        
        num_steps = len(train)
        metrics = []
        for step, (examples, labels) in enumerate(train):
            if examples.shape[0] == 1:
                examples = examples[0]
                labels = labels[0]
            original_labels = labels
            if not self.is_multiclass():
                labels = labels.argmax(dim=-1)

            skeleton_video.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
            logits, loss = model(examples, labels, tau=self.tau, reduction='avg')
            loss = loss.sum()
            loss.backward()
            optimizer.step()
            model.zero_grad()

            logits, prediction = self.activation(logits.float())
            tpr, tnr, nbac = NBAC(prediction, original_labels.float())
            auc = AUC(logits, original_labels.float())
            
            score = auc if self.hyper_params['conditions']['score_type'] == 'auc' else float(nbac.detach().float())
            metrics.append({
                'loss': loss.detach().float().cpu(),
                'score': score,
            })

            LOGGER.debug(
                '[train] [%02d] [%03d/%03d] loss:%.6f AUC:%.3f NBAC:%.3f tpr:%.3f tnr:%.3f, lr:%s',
                epoch, step, num_steps, loss, auc, nbac, tpr, tnr,
                [x['lr'] for x in optimizer.param_groups]
            )
        
        train_loss = np.average([m['loss'] for m in metrics])
        train_score = np.average([m['score'] for m in metrics])
        # optimizer.update(train_loss=train_loss)
        
        return {
            'loss': train_loss,
            'score': train_score,
        }

    def epoch_valid(self, epoch, valid, reduction='avg'):
        test_time_augmentation = False
        self.model.eval()
        num_steps = len(valid)
        metrics = []
        tau = self.tau

        with torch.no_grad():
            for step, (examples, labels) in enumerate(valid):
                original_labels = labels
                if not self.is_multiclass():
                    labels = labels.argmax(dim=-1)

                batch_size = examples.size(0)

                # Test-Time Augment flip
                # if self.use_test_time_augmentation and test_time_augmentation:
                #     examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)
                #     labels = torch.cat([labels, labels], dim=0)

                skeleton_video.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                logits, loss = self.model(examples, labels, tau=tau, reduction=reduction)

                # avergae
                # if self.use_test_time_augmentation and test_time_augmentation:
                #     logits1, logits2 = torch.split(logits, batch_size, dim=0)
                #     logits = (logits1 + logits2) / 2.0

                logits, prediction = self.activation(logits.float())
                tpr, tnr, nbac = NBAC(prediction, original_labels.float())
                if reduction == 'avg':
                    auc = AUC(logits, original_labels.float())
                else:
                    auc = max([AUC(logits[i:i+16], original_labels[i:i+16].float()) for i in range(int(len(logits)) // 16)])

                score = auc if self.hyper_params['conditions']['score_type'] == 'auc' else float(nbac.detach().float())
                metrics.append({
                    'loss': loss.detach().float().cpu(),
                    'score': score,
                })

                LOGGER.info(
                    '[valid] [%02d] [%03d/%03d] loss:%.6f AUC:%.3f NBAC:%.3f tpr:%.3f tnr:%.3f, lr:%s',
                    epoch, step, num_steps, loss, auc, nbac, tpr, tnr,
                    [x['lr'] for x in self.optimizer_all.param_groups]
                )
            if reduction == 'avg':
                valid_loss = np.average([m['loss'] for m in metrics])
                valid_score = np.average([m['score'] for m in metrics])
            elif reduction in ['min', 'max']:
                valid_loss = np.min([m['loss'] for m in metrics])
                valid_score = np.max([m['score'] for m in metrics])
            else:
                raise Exception('not support reduction method: %s' % reduction)


        return {
            'loss': valid_loss,
            'score': valid_score,
        }

    def skip_valid(self, epoch):
        LOGGER.debug('[valid] skip')
        return {
            'loss': 99.9,
            'score': epoch * 1e-4,
        }

    def prediction(self, dataloader, model=None, test_time_augmentation=True, detach=True, num_step=None):
        tau = self.tau
        if model is None:
            # model = self.model_pred
            model = self.model
            best_idx = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints]))
            best_loss = self.checkpoints[best_idx]['valid']['loss']
            best_score = self.checkpoints[best_idx]['valid']['score']

            # states = self.checkpoints[best_idx]['model']
            # model.load_state_dict(states)
            LOGGER.info('best checkpoints at %d/%d (valid loss:%f score:%f) tau:%f',
                        best_idx + 1, len(self.checkpoints), best_loss, best_score, tau)

        num_step = len(dataloader) if num_step is None else num_step
        
        model.eval()
        with torch.no_grad():
            predictions = []
            # for step, (examples, labels) in zip(range(num_step), dataloader):
            for examples, labels in dataloader:
                # LOGGER.info('pos 1, beginning')
                skeleton_video.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
                # LOGGER.info('pos 2, after moving to gpu')
                logits = model(examples, tau=tau)
                # LOGGER.info('pos 3, after model')
                logits, prediction = self.activation(logits)
                # LOGGER.info('pos 4, after activation')
                if detach:
                    predictions.append(logits.detach().float().cpu().numpy())
                else:
                    predictions.append(logits)
                # LOGGER.info('pos 5, ending')
            if detach:
                predictions = np.concatenate(predictions, axis=0).astype(np.float)
            else:
                predictions = torch.cat(predictions, dim=0)
        return predictions
