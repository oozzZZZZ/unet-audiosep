# Sound source separation with Deep U-net

ボーカル抽出のために作った音源分離モデル。<br>
音楽音声音源分離だけでなく、ノイズ除去などにも利用可能です。

ここではMUSDB18を使った音楽音源分離のための方法について書きます

## 環境構築
`sh setup.sh`

## step1
`parameter.py`より各種パスやパラメーターを指定する

`datatimes`で指定した数値倍に音声を増やします。<br>
`target` : 分離したい音声を指定する。<br>
`WINDOWS` : OSを選択してください。<br>

## step2
学習のための音声を生成し、学習を行います。<br>
Run `sh train.sh`<br>
学習曲線は`chack_loss.jpynb`で確認できます。

## step3
試聴方法<br>
`listen.jpynb`

# Reference
[Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017). Singing voice separation with deep u-net convolutional networks.](https://openaccess.city.ac.uk/id/eprint/19289/)

# Datasets
- [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)<br>
- [MedleyDB 2.0 Audio](https://medleydb.weebly.com/)
