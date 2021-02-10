#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:27:59 2020
@author: t.yamamoto
"""

import os
import numpy as np
import torch
import torch.utils.data as utils
from tqdm import tqdm
import random

from librosa.core import load,stft,resample
from librosa.util import find_files
from librosa.effects import pitch_shift, time_stretch

import parameter as C

def LoadAudio(fname):
    y, sr = load(fname, sr=None)
    if sr != C.SR:
       y = resample(y,sr,C.SR)
    spec = stft(y, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))
    return mag, phase

def LoadAudio_Arg(fname,pitch_shift,time_stretch):
    y, sr = load(fname, sr=C.SR)
    if sr != C.SR:
       y = resample(y,sr,C.SR)
    y = pitch_shift(y, C.SR, pitch_shift)
    y = time_stretch(y, time_stretch)
    spec = stft(y, n_fft=C.FFT_SIZE, hop_length=C.H, win_length=C.FFT_SIZE)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))
    return mag, phase

def LengthAdjuster(stft_data):
    if stft_data.shape[1] > C.PATCH_LENGTH:
        stft_data = stft_data[:,:C.PATCH_LENGTH]
    else: 
        while stft_data.shape[1] < C.PATCH_LENGTH:
            stft_data = np.concatenate((stft_data,stft_data),1)[:,:C.PATCH_LENGTH]
    return stft_data

def SaveSTFT():
    targetlist = find_files(C.target_path, ext="wav")
    noiselist = find_files(C.noise_path, ext="wav")
    noise_num = len(noiselist)
    target_index = 0
    for targetfile in tqdm(targetlist):

        target_mag, _ = LoadAudio(targetfile)
        norm = target_mag.max()
        skip_count = 0
        if target_mag.shape[0] >  C.PATCH_LENGTH:

            step = target_mag.shape[1] // C.PATCH_LENGTH

            for i in tqdm(range(step),leave = False):
                target_mag_p = target_mag[: , i*C.PATCH_LENGTH : (i+1)*C.PATCH_LENGTH]
                target_mag_p /= norm

                noise_file = noiselist[random.randint(0, noise_num-1)]
                noise_mag, _ = LoadAudio(noise_file)
                noise_mag = LengthAdjuster(noise_mag)
                noise_mag /= norm

                addnoise_mag = target_mag_p + noise_mag
                addnoise_mag /= norm
                
                fname = str(target_index) + "_" + str(i)
                np.savez(os.path.join(C.path_fft, fname+".npz"),speech=target_mag_p, addnoise=addnoise_mag)
                target_index += 1
        else:
            skip_count += 1

    print("SKIP:",skip_count)
    
def SaveSTFT_Arg(pitch_shift,time_stretch,argtime):
    targetlist = find_files(C.target_path, ext="wav")
    noiselist = find_files(C.noise_path, ext="wav")
    noise_num = len(noiselist)
    target_index = 0
    for targetfile in tqdm(targetlist):

        target_mag, _ = LoadAudio_Arg(targetfile)
        norm = target_mag.max()
        skip_count = 0
        if target_mag.shape[0] >  C.PATCH_LENGTH:

            step = target_mag.shape[0] // C.PATCH_LENGTH

            for i in tqdm(range(step),leave = False):
                target_mag_p = target_mag[: , i*C.PATCH_LENGTH : (i+1)*C.PATCH_LENGTH]
                target_mag_p /= norm

                noise_file = noiselist[random.randint(0, noise_num-1)]
                noise_mag, _ = LoadAudio(noise_file)
                noise_mag = LengthAdjuster(noise_mag)
                noise_mag /= norm

                addnoise_mag = target_mag_p + noise_mag
                addnoise_mag /= norm
                fname = str(target_index) + "_" + str(i) + str(argtime)
                np.savez(os.path.join(C.PATH_FFT, fname+"_arg.npz"),speech=target_mag_p, addnoise=addnoise_mag)
        else:
            skip_count += 1

    print("SKIP:",skip_count)
    
def use_data(data_list):
    num_data = len(data_list)
    a = round(num_data, -2)
    if a > num_data:  
        num_usedata = round(num_data-100, -2)
    else:
        num_usedata=a
    return num_usedata

def MyDataLoader():
    filelist = find_files(C.PATH_FFT, ext="npz")
    speech_trainlist = []
    addnoise_trainlist = []
    
    for file in tqdm(filelist,desc='[Loading..]'):
        ndata = np.load(file)    
        speech=torch.from_numpy(ndata["target"].astype(np.float32)).clone()
        addnoise=torch.from_numpy(ndata["data"].astype(np.float32)).clone()
        if speech.shape[1]==C.PATCH_LENGTH:
            speech_trainlist.append(speech)
            addnoise_trainlist.append(addnoise)
        
    train_num = use_data(speech_trainlist)
    
    tensor_speech_trainlist = torch.stack(speech_trainlist[:train_num])
    tensor_addnoise_trainlist = torch.stack(addnoise_trainlist[:train_num])
    
    print("Train dataset")
    print(">>Available data :", len(speech_trainlist))
    print(">>Use data :", train_num)
    
    traindataset = utils.TensorDataset(tensor_speech_trainlist,tensor_addnoise_trainlist)
    data_split = [int(0.2 * train_num),int(0.8 * train_num)]
    train_dataset,val_dataset = utils.random_split(traindataset,data_split)

    train_loader = utils.DataLoader(train_dataset,batch_size=C.BATCH_SIZE,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
    val_loader = utils.DataLoader(val_dataset,batch_size=C.BATCH_SIZE,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
    
    return train_loader,val_loader