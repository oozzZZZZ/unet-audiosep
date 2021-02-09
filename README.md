# unet_speech

音声の雑音除去するためのモデル
割と少ないデータ数でもなんとかなる。

ここではMUSDB18を使った音楽音源分離のための方法について書きます

## step1
`parameter.py`より各種パスやパラメーターを指定する

## step2
音声の事前処理を行います
Run `python3 preprocessing_musdb.py`

## step3
学習
Run `python3 train.py`
