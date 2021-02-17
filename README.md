# Sound source separation with Deep U-net

ボーカル抽出のために作った音源分離モデル。<br>
音楽音声音源分離だけでなく、ノイズ除去などにも利用可能です。

## 環境構築
macでしか動作確認していません。ターミナルで
```
git clone https://github.com/oozzZZZZ/unet-audiosep
cd unet-audiosep
sh setup.sh
```
pythonのデフォルトがpython2で動作しない時は、.shファイル内の`python`を`python3`に書き換えてください。
## 使い方
GUIからボーカル抽出できます。<br>

*起動方法*<br>
`sh run.sh`<br>
pythonのデフォルトがpython2で動作しない時は、.shファイル内の`python`を`python3`に書き換えてください。<br>
Epoch８０, Mask Rate85%くらいがおすすめですが，曲によって得手不得手あるようなのでいろいろ試してみてください。

* `If this fails your Python may not be configured for Tk`のようなエラーが発生する場合、仮想環境の不具合が影響しているようです。
* tkinterの再インストールを試すか、anaconda環境で実行してみてください。（anaconda推奨です） [pyenvのpythonでtkinterを使用する方法](https://qiita.com/skyloken/items/a5f839eba1bd79cd5ef9)

# Reference
[Jansson, A., Humphrey, E., Montecchio, N., Bittner, R., Kumar, A., & Weyde, T. (2017). Singing voice separation with deep u-net convolutional networks.](https://openaccess.city.ac.uk/id/eprint/19289/)

# Datasets
- [MUSDB18](https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems)<br>
- [MedleyDB 2.0 Audio](https://medleydb.weebly.com/)
