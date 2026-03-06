# 将棋における強さ推定器を用いた強さ調整AI

このレポジトリは、[Strength Estimation and Human-Like Strength Adjustment in Games](https://github.com/rlglab/strength-estimator)を基盤としています(README.mdだけなら[こちら](./docs/README.md)を確認してください)。
そのため、[MiniZero](https://github.com/rlglab/minizero)も基盤となっています。
また、将棋のルールには[sunfish3](https://github.com/sunfish-shogi/sunfish3)を利用しています。
以下に実験を再現するための手順を記述しています。

## 強さ推定器の学習

### 前提条件
本プログラムを動作させるには、Linuxオペレーティングシステムと、少なくとも1つのNVIDIA GPUが必要です。

### ビルド
まず、このレポジトリをクローンしてDockerfileを任意の名前でビルドします。(my-image-nameは任意に指定)
```bash
git clone --recursive git@github.com:rlglab/strength-estimator.git
docker build -t my-image-name -f ./minizero/Dockerfile .
cd strength-estimator
```
その後、作成したコンテナに入りビルドを行います。
```bash
# コンテナを実行
./scripts/start-container.sh --image my-image-name

# コンテナ内でプログラムをビルド
./scripts/build.sh shogi
```

### ゲーム情報の前処理
将棋の棋譜は[コンピュータ将棋対局場(floodgate)](http://wdoor.c.u-tokyo.ac.jp/shogi/x/)の棋譜倉庫から取得できます。

取得した棋譜の前処理を以下のように行い、trainingデータ、testデータ、candidateデータを作成します。
また、 $\texttt{rank}_{\infty}$ が必要な場合はrandomデータも作成します。
この時、以下のファイルの`csa_dir`(もしくは`csa_dirs`)に棋譜のある任意の場所を指定してください。
```bash
#棋譜の前処理
python3 parse_csa.py
python3 parse_csa_to_sgf.py

#rank_∞のデータ
python3 create_rankinf.py
```
なお、これらの処理はコンテナの外で実行してください。


### 強さ推定器の学習
強さ推定器の学習モデルを作成するためには、以下のコマンドを実行します。
```bash
./scripts/train.sh shogi cfg/se_shogi.cfg           #SE
./scripts/train.sh shogi cfg/se_infty_shogi.cfg     #SE_∞(rank_∞が含まれる)
```
学習を始めると以下の構造のフォルダが作成されます。
```bash
# 以下の例はSE_∞のものです
shogi_bt_b16_r9_p7_10bx256-2a0d91/
├── shogi_bt_b16_r9_p7_10bx256-2a0d91.cfg  #設定ファイル
├── model/                                     
│   ├── weight_iter_*.pkl                      #学習ステップ、パラメータ等
│   └── weight_iter_*.pt                       #モデルパラメータのみ（テスト用）
└── op.log                                     #ログ
```

## 評価

### 強さ推定のランク推定精度
以下のコマンドを実行してください。
```bash
./build/shogi/strength_shogi -conf_file cfg/se_shogi.cfg -mode evaluator        # SE
./build/shogi/strength_shogi -conf_file cfg/se_infty_shogi.cfg -mode evaluator  # SE_∞
```
コマンドを実行すると、以下のような表を出力します。
```
	0	1	2	3	4	5	6	7	all
1	0.756	0.312	0.216	0.332	0.286	0.238	0.214	0.388	0.34275
2	0.782	0.396	0.362	0.416	0.308	0.318	0.308	0.476	0.42075
3	0.83	0.52	0.374	0.49	0.348	0.402	0.378	0.56	0.48775
...
99	1	0.99	0.994	0.998	0.976	0.93	0.984	0.83	0.96275
100	1	0.996	0.992	0.99	0.986	0.934	0.986	0.834	0.96475
```

* 行：各行は特定のゲーム数における推定精度評価を表します。
* 列：ランク0から7の各ランクにおけるランク別の結果と、全ランクの平均推定精度評価を表します。

この時、設定ファイル(`cfg/se_infty_shogi.cfg`)の`nn_file_name`を変更することで、指定したモデルを使用できるようになります。

### 強さ調整・指し手一致率
強さ調整を行うためには、設定ファイル(`.cfg`)の以下の項目を以下のように変更してください。
```bash
# rank5(3500_3900 Eloレーティングも調整)の場合
testing_sgf_dir=rank_testing_txt/3500_3900.txt
candidate_sgf_dir=candidate_shogi/3500_3900
```
この変更を実施後、以下のコマンドを実行してください。
```bash
./build/shogi/strength_shogi -conf_file cfg/se_shogi_mcts.cfg -mode mcts_acc        # SE-MCTS
./build/shogi/strength_shogi -conf_file cfg/se_infty_shogi_mcts.cfg -mode mcts_acc  # SE_∞-MCTS
```
このコマンドを実行すると、強さ調整を行った後、以下の例のようにシミュレーション回数と様々なz値に対する指し手一致率の評価を出力します。
```
simulation: 1, mcts accuracy: 46.3942% (193/416), ssa_-2 accuracy: 46.3942% (193/416), ssa_-1 accuracy: 46.3942% (193/416), ...
simulation: 100, mcts accuracy: 38.4615% (160/416), ssa_-2 accuracy: 36.5385% (152/416), ssa_-1 accuracy: 37.0192% (154/416), ...
simulation: 200, mcts accuracy: 37.0192% (154/416), ssa_-2 accuracy: 36.7788% (153/416), ssa_-1 accuracy: 37.9808% (158/416), ...
...
```

* simulation：MCTSで用いられるシミュレーション回数
* mcts accuracy：SEモデルを用いたSE-MCTS精度
* ssa_* accuracy：異なるz値でSA-MCTSを用いた時の精度

### 対局
作成したモデルを実際にエンジンとして登録して対局させるには、ビルドした際に作成された`shogi_usi`を用います。
以下は[ShogiHome](https://sunfish-shogi.github.io/shogihome/)で実行させる手順を示します。

* 上記のリンクからShogiHomeをインストールしてください
* `shogi_bridge.sh`を確認し、使用する設定ファイル(`CONF_PATH`)を指定してください
* 確認したら、ShogiHomeで「エンジン設定」から「追加」を選択し、`shogi_bridge.sh`を選択する

また、デフォルトの設定ではランクが対局ごとに変動するようになっていますが、特定のランクと対局したい場合、`Rank`を0〜7の任意の値に設定してください(0が最も弱く、7が最も強いランク)。