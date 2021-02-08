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

learning_rate=0.0001
epochs=500

PATH_FFT = "./Uspec"

SPEECH_PATH="/disk107/Datasets/CMU_ARCTIC"
NOISE_PATH="/disk107/Datasets/noise/noise"

MODEL_PATH = "model"
