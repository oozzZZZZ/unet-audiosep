#!/usr/bin/bash
python -m venv unetspeech
. unetspeech/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
deactivate
