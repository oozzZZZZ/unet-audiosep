#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:36:13 2021

@author: t.yamamoto
"""

SR = 16000
H = 512
FFT_SIZE = 1024
BATCH_SIZE = 64
PATCH_LENGTH = 128

learning_rate=0.0005
epochs=400

PATH_FFT = "./stft_data" #データセットの生成場所

KEY_TYPE="./noise_data" #目的としない音声
UTT_PATH="./speech_data" #目的音声

MODEL_PATH = "model" # モデルの保存場所
