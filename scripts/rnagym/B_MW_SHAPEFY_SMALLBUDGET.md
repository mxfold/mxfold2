# B方向：MW / SHAPE-fy の短期replay比較（2026-10-11）

依頼：B方向も投入し、SHAPE-fyを検討する。
実行root：`qsub_logs/rnagym/rnagym-b-mw-shapefy-20261011`。
稼働中のA方向のコード・plan・ジョブは変更しない。

## 初期値とデータ

全条件を指定baseline
`/gs/bs/tga-satolab-gtex/yamauchi/TrainA_only_new_parameter/TrainSetA/epoch-100`
から開始する。既存MW事前学習checkpointはB方向の初期値にしない。
適応seedは1234/1235/1236であり、baselineの独立学習seedではない。

TrainSetA3166の構造例と固定RNAgym train SHAPE/2A3 3000 records
（2992 unique sequences）から、独立・復元抽出した1組の損失を加算する。
1組ずつ同じoptimizer updateに寄与させる方式であり、ランダムに
構造/chemical例を二者択一するreplay fraction方式ではない。
今回DMS、新しいfull-pool pretraining、RNAgym full testは含めない。

検証は既存TrainSetB1044と、train側から分離したSHAPE512。
SHAPE512はglobal edit identityのスクリーニング済みだが、family-disjointとは
主張しない。TestSetB430は除外・再分割しない。

## 共通条件と設定表

全層更新、AdamW、weight_decay=1e-4、clip_norm=1、構造FY perturb=.1、
score_loss_weight=0。A方向のbaseline再現条件とは異なり、過去の短期B実験に
合わせた設定である。LSTMのみnative Torch実装、CNNはcuDNN。
構造/化学の抽出generatorを独立化し、chemical側dropout/perturbationのRNGは
forkして構造側のstreamを変えない。全条件の構造露出は2560例。

| 条件 | lr | chemical係数 | accumulation | 最大updates |
|---|---|---|---|---|
| 構造のみ対照 | 1e-4 / 3e-5 | 0 | 1 | 2560 |
| MW | 1e-4 / 3e-5 | .003 | 1 | 2560 |
| SHAPE-fy | 1e-4 / 3e-5 | .5 | 1 | 2560 |
| MWの係数感度 | 1e-4 | .01 | 1 | 2560 |
| SHAPE-fyの係数感度 | 1e-4 | .05 | 1 | 2560 |
| accumulation8対照 | 1e-4 | 0 | 8 | 320 |
| SHAPE-fy accumulation8参照 | 1e-4 | .5 | 8 | 320 |

計10設定×3seed=30arms。acc8も2560組に上限を揃えるため、8倍のデータや
更新予算を与えない。acc1との比較は**露出を揃えたaccumulation比較**であり、
同じoptimizer更新数の比較ではない。SHAPE-fy acc8にも同じlr/accumulationの
構造のみ対照を置く。

MWはraw Mann–Whitney 1-AUROC、nu=.1、RMS、central、K=1、perturb=.1。
SHAPE-fyはREPORT.mdのslope=2.6、intercept=-.8、perturb=.1、追加score正則化0、
weight_schedule=noneを採用する。係数.5/lr1e-4/acc8はREPORTの損失・optimizer設定を
維持するが、replayあり、3000 pool、native LSTMのため、REPORTの実験の完全再現ではない。
SHAPE-fyは絶対reactivity値をpseudoenergyに変換する手法であり、MWと異なり
non-parametric評価の不変性を持つ損失ではない。両手法の係数の数値は直接比較しない。

## 検証・選択・評価

acc1はupdate0/100/300/512/1024/1536/2560、acc8は0/13/38/64/128/192/320で
TrainSetBとheldout SHAPE512を評価。acc8の実際のpair drawsは
0/104/304/512/1024/1536/2560である。共通baselineのupdate0評価は再利用する。
停止点はTrainSetB macro F1の最大、同値なら早いupdate。baseline0も候補。
SHAPE heldoutのloss/AUROC、degenerate率、paired率は副次的に追跡する。

全条件の選択を固定してから、選択checkpointと固定最終checkpointでTestSetBの
macro PPV/Sen/F1/MCCを評価する。baselineも直接評価する。同時に元のSHAPE3000の
MW loss/AUROCをperturb0/.1で計算する。選択と最終が同じ場合は評価を再利用する。
TestSetBの過去の反復利用による探索バイアスは解消されておらず、完全に未使用の
外部検証とは呼ばない。公式RNAgym test全件はこのパイロットには投入しない。

## 診断と安全策

GPU事前検証で、FY/係数0の完全一致、勾配診断有無の完全一致、MW/SHAPE-fyの
中断再開の完全一致、accumulation8の再開を確認する。さらに固定32組のtrain RNAで
MW/SHAPE-fyの初期raw勾配比・cosineを測定する。診断はoptimizer前処理・weight decayを
含まず、単一例のcosineやlossの大きさだけで優劣を判断しない。
訓練時にも100/300/1000update近辺の勾配と毎updateの成分lossを記録する。

各job gpu_h=1、訓練8h、評価2h、事前検証1h。いずれも24h以下。
訓練/評価arrayは同時3本。成功markerとsource/input hashを確認してから次段に進む。
UGEのhold_jidだけでは失敗を防げないため、依存元失敗時は処理を開始しない。
partial runを自動上書きしない。checkpointにはoptimizer・Torch/CUDA/Python RNG・
両sampling generatorを保存しており、必要時は監査して再開できる。

予測BPP/BPSEQは生成せず、RNAごとの結果はTSVに集約する。
checkpointは各arm最大6個。既存データを複製せずリストのみ再利用する。
全workflowの出力は数百ファイル程度であり、RNA数に比例したファイルは作らない。

投入job：preflight8967910、train8967911（30tasks）、select8967912、
evaluate8967913（baseline＋30tasks）、summarize8967914。

初期32組の診断では、両勾配が非ゼロの例に限定したraw auxiliary/structure
ノルム比中央値はMW9.11280、SHAPE-fy0.65006（係数をかける前）。
MW係数.003なら2.734%、.01なら9.113%、SHAPE-fy係数.05なら3.250%、
.5なら32.503%に相当する。非ゼロaux勾配はMW21/32、SHAPE-fy30/32。
係数の尺度差を確認でき、FYの弱い.05条件を含める根拠になるが、
この値はoptimizer更新比・精度・訓練中の持続性を保証しない。
全GPU smokeの完全一致assertionは通過し、baseline検証を計算中。
