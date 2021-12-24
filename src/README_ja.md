# プログラムの説明

## [my\_model](my_model/)
分析用のモデル
(PyTorchの公式参照)

下記のように "path.py" の
データセットのパスはハードコーディングなので
マシンごとに編集する．
* path.py
```
ILSVRC2012_DATASET_PATH = '/data1/dataset/ilsvrc2012'
```

## [train\_model](train_model/)
モデル訓練用のプログラム


## [utils](utils/)
下記のような用途をもった汎用プログラム集．

* 画像表示
* 画像の変換
* データセットの読み込み
* 受容野の算出
* 中間表現の追跡
* 設定の保存

## 直下プログラム

**共通する点**

* argparse で実験設定を決める． 
* 分析に用いた設定は [batch process](../batch_process/)を参照
* **大きな容量（100GB以上）になりやすいので，保存するデバイスには気をつける．**

--- 

* [random\_sample.txt](random_sample_val.txt)
  - 分析に使う小数のデータセット（ラベルの割合を考慮）

* [make\_rf\_datas.py](make_rf_datas.py)
  - 分析に使う中間ファイルの作成．
  - 主に受容野画像の切り出しと勾配情報の保存を行う．

* [analyize\_rf\_datas.py](analyize_rf_datas.py)
  - Top K個の受容野や平均受容野の算出を行う．
  - 詳しくは引数によって制御する．

* [analyize\_optimalinpu.py](analyize_optimalinput.py)
  - 活性値最大化法による可視化を行う．
  - Adam, L-BFGSに対応している．


* [analyize\_preact.py](analysis_preact.py)
  - あるニューロンの中間受容野（前の入力）を分析する．
  - 前入力は次元削減手法によって画像化する．

* [analyize\_sparse\_and\_svcca.py](analyize_sparse_and_svcca.py)
  - 分析する層はハードコーディング．
  - 活性値のスパースのカウントをグラフ化
  - SVCCAの値をグラフ化

* [analyize\_numberInClass.py](analyize_numberInClass.py)
  - ニューロンが好んだクラスの分析

* [train\_transfer\_model.py](train_transfer_model.py)
  - ResNetの転移学習を行う
  - 特定の中間層までを再学習することができる．

* [make\_image\_filter.py](make_image_filter.py)
  - 画像をYUV色空間で分析する．
  - 少し冗長でハードコーディングしている．

* [make\_analysis\_meanrfs.py](make_analysis_meanrfs.py)
  - 受容野画像のPCAを行う．
  - 少し冗長でハードコーディングしている．

**上記以外のプログラムの使用は非推奨**

同じような機能で，
冗長な表現で書かれたり，
ハードコーディングになっているものたち．

* receptive\_visualizer.py
* make\_topk\_image\_all.py
* etc
