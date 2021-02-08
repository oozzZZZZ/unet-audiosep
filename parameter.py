#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:36:13 2021

@author: t.yamamoto
"""
SPEECH_PATH="/speech_dir" #target signal
NOISE_PATH="/noise_dir" #noise signal

MODEL_PATH = "model" # save model
PATH_FFT = "./dataset_dir" # save dataset

datatimes = 10 #Increase the data n times

SR = 16000
H = 512
FFT_SIZE = 1024
BATCH_SIZE = 64
PATCH_LENGTH = 128

learning_rate=0.0001
epochs=500
