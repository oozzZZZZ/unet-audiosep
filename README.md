# Sound source separation with Deep U-net

ボーカル抽出のために作った音源分離モデル。<br>
音楽音声音源分離だけでなく、ノイズ除去などにも利用可能です。

## 環境構築
```
git clone https://github.com/oozzZZZZ/unet-audiosep
cd unet-audiosep
sh setup.sh
```

## 使い方
GUIから分離音声を選択できます。<br>
`sh run.sh`

# Reference
[Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017). Singing voice separation with deep u-net convolutional networks.](https://openaccess.city.ac.uk/id/eprint/19289/)

# Datasets
- [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)<br>
- [MedleyDB 2.0 Audio](https://medleydb.weebly.com/)
