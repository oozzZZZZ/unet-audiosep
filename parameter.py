#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:36:13 2021

@author: t.yamamoto
"""
MODEL_PATH = "D:/yamamoto/model_musdb" # save model
PATH_FFT = "D:/yamamoto/musdb_stft_dataset" # save dataset

MUSDB_PATH = "D:/yamamoto/音源分離用データ/MUSDB" # musdb18 dir path
target = "vocals" # vocals/drums/bass/other separation target

WINDOWS = True
# Select your OS: Windows=>True / mac or linux=>False

datatimes = 10 #Increase training data n times

SR = 16000
H = 512
FFT_SIZE = 1024
BATCH_SIZE = 64
PATCH_LENGTH = 128

learning_rate=0.0001
epochs=600

#datasplit
train,val = 0.8, 0.2

pre_trained = False
pre_trained_model = "./model/model_20210208_060155.pt" #事前学習モデル
