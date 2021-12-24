# プログラムの説明

**コードの分け方はテキトー**

## 直下プログラム

### 分析補助

* analysis.py 
  - よく使う分析まわりの関数

* config.py 
  - 実験設定の再現性を保つため
  - argparseなどの設定情報を保存・読み込みする関数
 
* dumpLoaders.py 
  - make\_rf\_datas.py で作成した中間ファイルの読み込みに用いる．
 
* my\_parser.py 
  - よく使う argparse を決める 
 
* performance\_model.py 
  - 正答率の評価を行う

### Deep用

* get\_dataset.py 
  - 特定のデータセットを取得する
 
* imagenet1000\_classname.py 
  - ImageNet1000のクラスを返す
 
* load\_model.py 
  - 特定のモデルを習得する
  - 学習済みモデルなども対応している
 
* tensortracker.py 
  - PyTorch で作成した中間表現を追跡することができる． 

* receptive\_field\_tracker.py 
  - 特定のニューロンの受容野を算出する．
  - ReceptiveFieldHook.py と tensortracker.py を合わせたもの．
 
* receptive\_field.py 
  - 受容野の切り出しに関する操作
 
* reset\_weight.py 
  - 特定の層の重みを初期化する

### 画像系

* colors.py 
  - 色空間の変換など

* plots.py 
  - 画像の表示まわりの処理


**上記以外のプログラムの使用は非推奨**

* ReceptiveFieldHook.py 

* dataloader\_ilsvrc2012.py 

* test\_pytorch-train.py 

* sample\_image\_imagenet.py 

* remove\_duplicate\_info.py 
