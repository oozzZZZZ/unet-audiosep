#!/usr/bin/bash
. unetspeech/bin/activate
python preprocessing_musdb.py
python train.py
deactivate
