# unet_speech

音声の雑音除去するためのモデル
割と少ないデータ数でもなんとかなる。

ここではMUSDB18を使った音楽音源分離のための方法について書きます

## step1
`parameter.py`より各種パスやパラメーターを指定する

```
MODEL_PATH = "D:/yamamoto/model_musdb" # save model
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
音声の事前処理を行います
Run `python3 preprocessing_musdb.py`

各トラックの音声をシャッフルし、毎回新たな曲を適当に生成しています。
回すたびに`datatimes`倍音声を増やすので、足りなければ都度実行してください。
今回はMUSDBの訓練データ、検証データ関係なく全てのデータを使って学習します。
ベンチマークにはほかのデータをご用意ください。
ちょっと聞く分にはデータセット内のミックス音声を使ってやってください。

## step3
学習
Run `python3 train.py`

学習中の損失曲線はjupyter notebook`check_loss`で確認できます。
`MODEL_PATH`に指定したディレクトリ内から最新の`loss_ ~ .npz`ファイルを選択し実行してください。

## step4
試聴

工事中
