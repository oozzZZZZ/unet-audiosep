#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:07:45 2021

@author: t.yamamoto
"""
import os
from tqdm import tqdm
import random
import numpy as np
import datetime

from librosa.util import find_files
from librosa.core import load,stft,resample

import parameter as C

PATH_FFT = C.PATH_FFT
SPEECH_PATH = C.SPEECH_PATH
NOISE_PATH = C.NOISE_PATH

times = C.datatimes

for time in tqdm(range(times)):

    speechlist = find_files(SPEECH_PATH, ext="wav")
    noiselist = find_files(NOISE_PATH, ext="wav")

    random.shuffle(speechlist)
    random.shuffle(noiselist)

    if not os.path.exists(PATH_FFT):
        os.mkdir(PATH_FFT)
    noise_num = len(noiselist)

    for i in tqdm(range(len(speechlist)-1),leave=False):
        
        target, sr = load(speechlist[i], sr=None)
        if sr != C.SR:
            target = resample(target, sr, C.SR)
            
        #speech data
        spec = stft(target, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
        fulllen=spec.shape[1]

        while fulllen<C.PATCH_LENGTH * (C.BATCH_SIZE+1) :
            i+=1
            target, sr = load(speechlist[i], sr=None)
            if sr != C.SR:
                target = resample(target, sr, C.SR)
            conc = stft(target, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
            spec = np.concatenate((spec,conc),1)
            fulllen = spec.shape[1]
        speech_spec = spec[:C.PATCH_LENGTH * (C.BATCH_SIZE+1) ]

        #noise data
        noise, sr = load(noiselist[[random.randint(0, noise_num-1)]], sr=None)
        if sr != C.SR:
            noise = resample(noise, sr, C.SR)
        spec = stft(noise, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
        fulllen=spec.shape[1]

        while fulllen<C.PATCH_LENGTH * (C.BATCH_SIZE+1) :
            i+=1
            noise, sr = load(noiselist[[random.randint(0, noise_num-1)]], sr=None)
            if sr != C.SR:
                noise = resample(noise, sr, C.SR)
            conc = stft(noise, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
            spec = np.concatenate((spec,conc),1)
            space = np.zeros([spec.shape[0],random.randint(1,120)])
            spec = np.concatenate((spec,space),1)
            fulllen = spec.shape[1]
        noise_spec = spec[:C.PATCH_LENGTH * (C.BATCH_SIZE+1)]

        #data mixer
        speech_spec=speech_spec[:,: C.PATCH_LENGTH * C.BATCH_SIZE]
        noise_spec=noise_spec[:,: C.PATCH_LENGTH * C.BATCH_SIZE]
        mix_spec=speech_spec+noise_spec

        speech_mag = np.abs(speech_spec)
        speech_mag /= np.max(speech_mag)
        mix_mag = np.abs(mix_spec)
        mix_mag /= np.max(mix_mag)
            
        iterate = 0
        now = datetime.datetime.now()
        while length > C.PATCH_LENGTH:
            save_target = speech_mag[:,C.PATCH_LENGTH * iterate:C.PATCH_LENGTH * (iterate+1)]
            save_data = mix_mag[:,C.PATCH_LENGTH * iterate:C.PATCH_LENGTH * (iterate+1)]
            fname = now.strftime('%Y%m%d%H%M%S') + "_" + str(iterate)
            np.savez(os.path.join(C.PATH_FFT, fname+".npz"),target=save_target, data=save_data)
            length-=PATCH_LENGTH
            iterate+=1
