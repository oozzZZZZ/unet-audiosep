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
`datatimes`で指定した数値倍に音声を増やします。

## step2
音声の事前処理を行います
Run `python3 preprocessing_musdb.py`

回すたびに`datatimes`倍音声を増やすので、足りなければ都度実行してください。
今回はMUSDBの訓練データ、検証データ関係なく全てのデータを使って学習します。
ベンチマークにはほかのデータをご用意ください。

## step3
学習
Run `python3 train.py`

学習中の損失曲線はjupyter notebook`check_loss`で確認できます。
`MODEL_PATH`に指定したディレクトリ内から最新の`loss_ ~ .npz`ファイルを選択し実行してください。

## step4
試聴

工事中
