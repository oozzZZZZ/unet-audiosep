# unet_speech

音声の雑音除去するためのモデル
割と少ないデータ数でもなんとかなる。

ここではMUSDB18を使った音楽音源分離のための方法について書きます

## 環境構築
`sh setup.sh`

## step1
`parameter.py`より各種パスやパラメーターを指定する

```
MODEL_PATH = "D:/yamamoto/model_musdb" # dir for save model
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

pre_trained_model = "./model/model_20210208_060155.pt" #事前学習モデル
```
`datatimes`で指定した数値倍に音声を増やします。<br>
`target` : 分離したい音声を指定する。

## step2
学習のための音声を生成し、学習を行います。<br>
Run `sh run.sh`

## step3
試聴方法<br>
`listen.jpynb`

## 参考文献
[Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017). Singing voice separation with deep u-net convolutional networks.](https://openaccess.city.ac.uk/id/eprint/19289/)
