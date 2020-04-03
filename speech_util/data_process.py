#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by HazzaCheng on 2019-09-22
import os
from functools import partial
from multiprocessing.pool import ThreadPool

import librosa
import numpy as np
from tensorflow.python.keras.preprocessing import sequence

from .CONSTANT import NUM_MFCC, FFT_DURATION, HOP_DURATION
from .tools import timeit, log
from .sonopy import *
pool = ThreadPool(os.cpu_count())



import torchaudio
import torch
import math
from torchaudio import functional as F
import tensorflow as tf

from .sonopy import *
def ohe2cat(label):
    return np.argmax(label, axis=1)

def torch_todb(x):
    multiplier=10.0
    amin=1e-10
    ref_value=1.0
    db_multiplier=math.log10(max(amin,ref_value))
    top_db=80
    x_db=multiplier*torch.log10(torch.clamp(x,min=amin))
    x_db-=multiplier*db_multiplier
    x_db=x_db.clamp(min=x_db.max().item()-top_db)
    return x_db
#@timeit
def get_max_length(x, ratio=0.95):
    """
    Get the max length cover 95% data.
    """
#     print(type(x))
#     print(x.shape)
#     print(x[0])
    lens = [len(_) for _ in x]
    max_len = max(lens)
    min_len = min(lens)
    lens.sort()
    # TODO need to drop the too short data?
    specified_len = lens[int(len(lens) * ratio)]
    log("Max length: %d; Min length %d; 95 length %d" % (max_len, min_len, specified_len))
    return specified_len


def pad_seq(data, pad_len):
    return sequence.pad_sequences(data, maxlen=pad_len, dtype='float32', padding='post', truncating='post')


def extract_parallel(data, extract):
    data_with_index = list(zip(data, range(len(data))))
    results_with_index = list(pool.map(extract, data_with_index))

    results_with_index.sort(key=lambda x: x[1])

    results = []
    for res, idx in results_with_index:
        results.append(res)

    return np.asarray(results)

# mfcc
#@timeit
def extract_mfcc(data, sr=16000, n_mfcc=NUM_MFCC):
    results = []
    for d in data:
        r = librosa.feature.mfcc(d, sr=sr, n_mfcc=n_mfcc)
        r = r.transpose()
        results.append(r)

    return results


def extract_for_one_sample(tuple, extract, use_power_db=False, **kwargs):
    data, idx = tuple
    r = extract(data, **kwargs)
    # for melspectrogram
    if use_power_db:
        r = librosa.power_to_db(r)

    r = r.transpose()
    return r, idx
#@timeit
def get_max_once(a,wl):
    t = torch.cuda.get_device_properties(0).total_memory
    c = torch.cuda.memory_cached(0)
    safe= 2147483648
#     with tf.Session() as sess:
#         tfused=sess.run(tf.contrib.memory_stats.BytesInUse())
#         maxused=sess.run(tf.contrib.memory_stats.MaxBytesInUse())
#     max_once=int((t-c-maxused)/80/wl)
#     print("gpu : {}(torch),{}:{}(tf) -> {}  using {} samples of {} in all with wl {}".format(c,tfused,maxused,t,max_once,a,wl))
    max_once=int((t-c-safe)/80/wl)
    print("gpu : {}(torch) -> {}  using {} samples of {} in all with wl {} and safe {}".format(c,t,max_once,a,wl,safe))
    return max_once
# #@timeit
# def extract_mfcc_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mfcc=NUM_MFCC):
#     if n_fft is None:
#         n_fft = int(sr*FFT_DURATION)
#     if hop_length is None:
#         hop_length = int(sr*HOP_DURATION)

#     batch,wl=data.shape
#     torch_mel=torchaudio.transforms.MelSpectrogram(sample_rate=sr,n_fft=n_fft,hop_length=hop_length).cuda()
# #     torch_todb=torchaudio.transforms.AmplitudeToDB(top_db=80).cuda()
#     dct_mat=F.create_dct(n_mfcc,torch_mel.n_mels,'ortho').cuda()
#     max_once=min(get_max_once(batch,wl),batch)
#     with torch.no_grad():
#         results_g=torch_todb(torch_mel(torch.FloatTensor(data[:max_once]).cuda()))
#         results_g=torch.matmul(results_g.transpose(1,2),dct_mat).transpose(1,2)
#         results_g=results_g.reshape(torch.Size([max_once])+results_g.shape[-2:])
# #         show_gpu(max_once,batch)
#         get_max_once(batch,wl)
#         results=results_g.cpu().numpy()
#     del results_g
#     torch.cuda.empty_cache()
#     cnt=max_once
#     while cnt<batch:  
#         print("mel batch cal {} -> {}".format(cnt,batch))
#         max_once=min(max_once,batch-cnt)
#         with torch.no_grad():
#             results_g=torch_todb(torch_mel(torch.FloatTensor(data[cnt:cnt+max_once]).cuda()))
#             results_g=torch.matmul(results_g.transpose(1,2),dct_mat).transpose(1,2)
#             results_g=results_g.reshape(torch.Size([max_once])+results_g.shape[-2:])
#             results=np.concatenate([results,results_g.cpu().numpy()],axis=0)
#         del results_g
#         torch.cuda.empty_cache()
#         cnt+=max_once
#     return results

#@timeit
def extract_mfcc_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mfcc=NUM_MFCC):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.mfcc, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
    results = extract_parallel(data, extract)

    return results


# zero crossings

#@timeit
def extract_zero_crossing_rate_parallel(data):
    extract = partial(extract_for_one_sample, extract=librosa.feature.zero_crossing_rate, pad=False)
    results = extract_parallel(data, extract)

    return results


# spectral centroid

#@timeit
def extract_spectral_centroid_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_centroid, sr=sr,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results
g_sr=16000
g_n_fft = int(g_sr*FFT_DURATION)
g_hop_length= int(g_sr*HOP_DURATION)
g_n_mels=30
g_torch_mel=torchaudio.transforms.MelSpectrogram(sample_rate=g_sr,n_fft=g_n_fft,hop_length=g_hop_length,n_mels=g_n_mels).cuda()
# @timeit
def extract_melspectrogram_parallel(data, sr=16000, n_fft=None, hop_length=None, n_mels=40, use_power_db=False):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    
    batch,wl=data.shape
    max_once=get_max_once(batch,wl)
    if max_once<=0:
        return extract_melspectrogram_parallel_cpu(data,sr,n_fft,hop_length,n_mels,use_power_db)
#     torch_mel=torchaudio.transforms.MelSpectrogram(sample_rate=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels).cuda()
    torch_mel=g_torch_mel
    with torch.no_grad():
        results_g=torch_todb(torch_mel(torch.FloatTensor(data[:max_once]).cuda()))
        results=results_g.cpu().numpy()
    del results_g
    torch.cuda.empty_cache()
    cnt=max_once
    while cnt<batch:  
#         print("mel batch cal {} -> {}".format(cnt,batch))
        with torch.no_grad():
            results_g=torch_todb(torch_mel(torch.FloatTensor(data[cnt:cnt+max_once]).cuda()))
            results=np.concatenate([results,results_g.cpu().numpy()],axis=0)
        del results_g
        torch.cuda.empty_cache()
        cnt+=max_once
    return results
# @timeit
def extract_melspectrogram_parallel_cpu(data, sr=16000, n_fft=None, hop_length=None, n_mels=40, use_power_db=False):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    
    torch_mel=torchaudio.transforms.MelSpectrogram(sample_rate=sr,n_fft=n_fft,hop_length=hop_length,n_mels=n_mels)
    with torch.no_grad():
        results_g=torch_todb(torch_mel(torch.FloatTensor(data)))
    results=results_g.numpy()
    return results
# @timeit
def extract_melspectrogram_parallel_librosa(data, sr=16000, n_fft=None, hop_length=None, n_mels=40, use_power_db=False):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.melspectrogram,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, use_power_db=use_power_db)
    results = extract_parallel(data, extract)

    return results

def extract_for_one_sample2(tuple, extract, use_power_db=False, **kwargs):
    data, idx = tuple
    r = extract(data, **kwargs)
    # for melspectrogram
    if use_power_db:
        r = librosa.power_to_db(r)
    return r, idx

# @timeit
def extract_melspectrogram_parallel_sonopy(data, sr=16000, n_fft=None, hop_length=None, n_mels=40, use_power_db=False):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample2,extract=mel_spec,sample_rate=sr, window_stride=(hop_length, hop_length),num_filt=n_mels,fft_size=n_fft)
    results = extract_parallel(data, extract)
    return results
# spectral rolloff
#@timeit
def extract_spectral_rolloff_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_rolloff,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)  # data+0.01?
    # sklearn.preprocessing.scale()
    return results


# chroma stft
#@timeit
def extract_chroma_stft_parallel(data, sr=16000, n_fft=None, hop_length=None, n_chroma=12):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_stft, sr=sr,
                      n_fft=n_fft, hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


#@timeit
def extract_bandwidth_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_bandwidth,
                      sr=sr, n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


#@timeit
def extract_spectral_contrast_parallel(data, sr=16000, n_fft=None, hop_length=None, n_bands=6):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_contrast,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands)
    results = extract_parallel(data, extract)

    return results


#@timeit
def extract_spectral_flatness_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.spectral_flatness,
                      n_fft=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


#@timeit
def extract_tonnetz_parallel(data, sr=16000):
    extract = partial(extract_for_one_sample, extract=librosa.feature.tonnetz, sr=sr)
    results = extract_parallel(data, extract)
    return results


#@timeit
def extract_chroma_cens_parallel(data, sr=16000, hop_length=None, n_chroma=12):
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)
    extract = partial(extract_for_one_sample, extract=librosa.feature.chroma_cens, sr=sr,
                      hop_length=hop_length, n_chroma=n_chroma)
    results = extract_parallel(data, extract)

    return results


#@timeit
def extract_rms_parallel(data, sr=16000, n_fft=None, hop_length=None):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.rms,
                      frame_length=n_fft, hop_length=hop_length)
    results = extract_parallel(data, extract)

    return results


#@timeit
def extract_poly_features_parallel(data, sr=16000, n_fft=None, hop_length=None, order=1):
    if n_fft is None:
        n_fft = int(sr*FFT_DURATION)
    if hop_length is None:
        hop_length = int(sr*HOP_DURATION)

    extract = partial(extract_for_one_sample, extract=librosa.feature.poly_features,
                      sr=sr, n_fft=n_fft, hop_length=hop_length, order=order)
    results = extract_parallel(data, extract)

    return results