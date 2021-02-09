# unet_speech

音声の雑音除去するためのモデル
割と少ないデータ数でもなんとかなる。

ここではMUSDB18を使った音楽音源分離のための方法について書きます

## step1
`parameter.py`より各種パスやパラメーターを指定する

必要項目
```
MODEL_PATH
PATH_FFT
MUSDB_PATH

datatimes = 10 #Increase the data n times

SR = 16000
H = 512
FFT_SIZE = 1024
BATCH_SIZE = 64
PATCH_LENGTH = 128

learning_rate=0.0001
epochs=500
```

## step2
音声の事前処理を行います
Run `python3 preprocessing_musdb.py`

## step3
学習
Run `python3 train.py`
