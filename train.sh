#!/usr/bin/bash
. unetspeech/bin/activate
cat pikachu.ansi
python preprocessing_musdb.py
python train_plus.py
