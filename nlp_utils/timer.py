# -*- coding: utf-8 -*-
from __future__ import absolute_import
import time
from collections import OrderedDict
import logging


LOGGER = logging.getLogger(__name__)

class stampTimer:
    def __init__(self):
        self.times = [time.time()]
        self.accumulation_time = OrderedDict({})
        self.accumulation = OrderedDict({})
        self.total_time = 0.0
        self.step_time = 0.0
    
    def get_latest_delta(self, name='GLOBAL'):
        if name == 'GLOBAL':
            return self.times[-1] - self.times[-2]
        return self.accumulation_time[name][-1] - self.accumulation_time[name][-2]

    def __call__(self, name, exclude_total=False, exclude_step=False, reset_step=False):
        self.times.append(time.time())
        delta = self.times[-1] - self.times[-2]

        if name not in self.accumulation:
            self.accumulation_time[name] = [self.times[-1]]
            delta_a = 0
        else:
            self.accumulation_time[name].append(self.times[-1])
        
        self.accumulation[name] = self.accumulation_time[name][-1] - self.accumulation_time[name][0]
        
        if not exclude_total:
            self.total_time += delta

        if reset_step:
            self.step_time = 0.0
        elif not exclude_step:
            self.step_time += delta

        return delta

    def __repr__(self):
        results = []
        for key, value in self.accumulation.items():
            results.append('{0}={1:.3f}'.format(key, value))
        return self.__class__.__name__ + '(total={0}, step={1}, {2})'.format(
            self.total_time, self.step_time, ', '.join(results)
        )

class Timer:
    def __init__(self):
        self.times = [time.time()]
        self.accumulation = OrderedDict({})
        self.total_time = 0.0
        self.step_time = 0.0

    def __call__(self, name, exclude_total=False, exclude_step=False, reset_step=False):
        self.times.append(time.time())
        delta = self.times[-1] - self.times[-2]

        if name not in self.accumulation:
            self.accumulation[name] = 0.0
        self.accumulation[name] += delta

        if not exclude_total:
            self.total_time += delta

        if reset_step:
            self.step_time = 0.0
        elif not exclude_step:
            self.step_time += delta

        return delta

    def __repr__(self):
        results = []
        for key, value in self.accumulation.items():
            results.append('{0}={1:.3f}'.format(key, value))
        return self.__class__.__name__ + '(total={0}, step={1}, {2})'.format(
            self.total_time, self.step_time, ', '.join(results)
        )
