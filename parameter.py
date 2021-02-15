#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:36:13 2021

@author: t.yamamoto
"""
MODEL_PATH = "D:/yamamoto/model_musdb" # save model
PATH_FFT = "D:/yamamoto/musdb_stft_dataset" # save dataset

#If you have MUDSB18
MUSDB_PATH = "D:/yamamoto/音源分離用データ/MUSDB" # musdb18 dir path
target = "vocals" # vocals/drums/bass/other separation target

#If you use your dataset...
TARGET_PATH = "./speech" #wav
NOISE_PATH = "./other" #wav

WINDOWS = True
# Select your OS: Windows=>True / mac or linux=>False

augmentation = True
"""
If augmentation = True     
Do Amplitude Shift, Audio Stretch, Time Shift,Pitch Shift
It takes time to process the data.
"""
datatimes = 10 #Increase training data n times

SR = 16000
H = 512
FFT_SIZE = 1024
BATCH_SIZE = 64
PATCH_LENGTH = 128

learning_rate=0.0001
epochs=600
save_epoch=20

#datasplit
train,val = 0.8, 0.2 # train + val = 1.0

pre_trained = False
pre_model_path = "./model/model_20210208_060155.pt" #事前学習モデル
