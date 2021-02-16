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
from scipy.io.wavfile import write
from librosa.core import load,stft,resample,istft
from librosa.util import find_files
from librosa.effects import pitch_shift, time_stretch

import parameter as C
import network

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
        
        if not (torch.isnan(speech).any() or torch.isnan(addnoise).any()):
            
            if speech.shape[1]==C.PATCH_LENGTH:
                speech_trainlist.append(speech)
                addnoise_trainlist.append(addnoise)
        
    train_num = use_data(speech_trainlist)
    
    tensor_speech_trainlist = torch.stack(speech_trainlist[:train_num])
    tensor_addnoise_trainlist = torch.stack(addnoise_trainlist[:train_num])
    
    print("Dataset")
    print(" >>Available data :", len(speech_trainlist))
    print(" >>Use data :", train_num)
    
    traindataset = utils.TensorDataset(tensor_speech_trainlist,tensor_addnoise_trainlist)
    data_split = [int(C.train * train_num),int(C.val * train_num)]
    train_dataset,val_dataset = utils.random_split(traindataset,data_split)
    print("\nTrain dataset",len(train_dataset),"\nVal Dataset",len(val_dataset))
    
    if C.WINDOWS:
        # Windowsではこっち
        train_loader = utils.DataLoader(train_dataset,batch_size=C.BATCH_SIZE,pin_memory=True,shuffle=True)
        val_loader = utils.DataLoader(val_dataset,batch_size=C.BATCH_SIZE,pin_memory=True,shuffle=True)
    else:
        # Mac, Linuxではこっち
        train_loader = utils.DataLoader(train_dataset,batch_size=C.BATCH_SIZE,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
        val_loader = utils.DataLoader(val_dataset,batch_size=C.BATCH_SIZE,num_workers=os.cpu_count(),pin_memory=True,shuffle=True)
    
    return train_loader,val_loader

def denoiser(audio_data,model_path,hard_rate=0.9):

    
    SR = 16000
    H = 512
    FFT_SIZE = 1024
    BATCH_SIZE = 64
    PATCH_LENGTH = 128

    model = network.UnetConv2()
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

    spec = stft(audio_data, n_fft=FFT_SIZE, hop_length=H, win_length=FFT_SIZE)

    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))

    full_len=mag.shape[1]
    times, mod = divmod(full_len, PATCH_LENGTH * BATCH_SIZE)

    if not times==0:
        for i in range(times):
            process = mag[:,PATCH_LENGTH * BATCH_SIZE * i:PATCH_LENGTH * BATCH_SIZE * (i+1)]
            listen_list=[]
            for iterate in range(BATCH_SIZE):
                _data = process[:,PATCH_LENGTH * iterate:PATCH_LENGTH * (iterate+1)]
                _data=torch.from_numpy(_data.astype(np.float32)).clone()
                listen_list.append(_data)
            tensor_data = torch.stack(listen_list)
            mask=model(tensor_data)
            mask[mask < hard_rate]=0
            h=tensor_data * mask
            h = h.to('cpu').detach().numpy().copy()
            mask_inst=1-mask
            inst=tensor_data * mask_inst
            inst = inst.to('cpu').detach().numpy().copy()

            if i==0:
                output = h[0,:,:]
                output_inst = inst[0,:,:]
                for f in range(1,BATCH_SIZE):
                    output = np.concatenate([output, h[f,:,:]], 1)
                    output_inst = np.concatenate([output_inst, inst[f,:,:]], 1)

            else:
                for f in range(BATCH_SIZE):
                    output = np.concatenate([output, h[f,:,:]], 1)
                    output_inst = np.concatenate([output_inst, inst[f,:,:]], 1)

        if not mod==0:            
            process = mag[:,-mod:]
            addempty=np.zeros([mag.shape[0],PATCH_LENGTH * BATCH_SIZE - mod])
            process = np.concatenate([process,addempty], 1)

            listen_list=[]
            for iterate in range(BATCH_SIZE):
                _data = process[:,PATCH_LENGTH * iterate:PATCH_LENGTH * (iterate+1)]
                _data=torch.from_numpy(_data.astype(np.float32)).clone()
                listen_list.append(_data)
            tensor_data = torch.stack(listen_list)
            mask=model(tensor_data)
            mask[mask < hard_rate]=0
            h=tensor_data * mask
            h = h.to('cpu').detach().numpy().copy()
            mask_inst=1-mask
            inst=tensor_data * mask_inst
            inst = inst.to('cpu').detach().numpy().copy()
            for f in range(BATCH_SIZE):
                output = np.concatenate([output, h[f,:,:]], 1)
                output_inst = np.concatenate([output_inst, inst[f,:,:]], 1)

    else:
        process = mag[:,-mod:]
        addempty=np.zeros([mag.shape[0],PATCH_LENGTH * BATCH_SIZE - mod])
        process = np.concatenate([process,addempty], 1)

        listen_list=[]
        for iterate in range(BATCH_SIZE):
            _data = process[:,PATCH_LENGTH * iterate:PATCH_LENGTH * (iterate+1)]
            _data=torch.from_numpy(_data.astype(np.float32)).clone()
            listen_list.append(_data)
        tensor_data = torch.stack(listen_list)
        mask=model(tensor_data)
        mask/=torch.max(mask)
        mask[mask < hard_rate]=0
        h=tensor_data * mask
        h = h.to('cpu').detach().numpy().copy()
        
        mask_inst=1-mask
        inst=tensor_data * mask_inst
        inst = inst.to('cpu').detach().numpy().copy()
        
        output_inst = inst[0,:,:]
        output = h[0,:,:]
        for f in range(1,BATCH_SIZE):
            output = np.concatenate([output, h[f,:,:]], 1)
            output_inst = np.concatenate([output_inst, inst[f,:,:]], 1)
            
    denoise=istft(output[:,:phase.shape[1]]*phase,hop_length=H, win_length=FFT_SIZE)
    inst=istft(output_inst[:,:phase.shape[1]]*phase,hop_length=H, win_length=FFT_SIZE)

    return denoise,inst

def separation_main(filename,model,maskrate,vocalpath,instpath):
    model_path = './model/model/'+model+'.pt'
    data, sr = load(filename, sr=16000)
    vocal,inst = denoiser(data,model_path,hard_rate=maskrate)
    write(vocalpath, 16000, vocal/np.max(vocal))
    write(instpath, 16000, inst/np.max(inst))