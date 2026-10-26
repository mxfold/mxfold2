# eFold公開データによるMXfold2学習

2026-10-26の追加指示により、3手法を他スレッドのAdaBelief復元系列に揃えて
`qsub_logs/efold_hf/bprna10_restored_ft5_replay20_20261026/` で再実験する。
モデルに加えてoptimizerのmoments/stepをbpRNA epoch10から復元し、
lr=1e-4、weight decay=1e-4、蓄積4、chemical係数.5、Turner固定/正則化.125。
MWのDMS A/C層別化、複数probeの目的集約、SHAPE-fy、構造replayは
参照元の凍結loss実装を使う。FT5epoch、replay20%、各epoch評価。
詳細と照合根拠は新rootの `README_ja.md`、`plan.json`、`evidence/` を参照。
旧fresh optimizerのMWジョブ9150625は停止した。以下は初回実験の履歴。

実験ルート:
`qsub_logs/efold_hf/bprna100_ft5_replay20_20261024/`

2026-10-24の指定変更により、事前学習は既存チェックポイントから継続し、
10エポック完了時にFTへ進む。実験ディレクトリ名は初回作成時のものを維持する。

`plan.json` にデータのHugging Faceリビジョン、SHA256、除外件数、
学習設定、固定した実装のハッシュを保存する。各ジョブは実験ルートの
`source/` を使い、作業中のリポジトリ変更から独立して動く。

## データと学習条件

[著者のデータ処理コード](https://github.com/rouskinlab/efold_data) と
[Hugging Face公開データ](https://huggingface.co/rouskinlab) を使用。
公開 `efold_train` に採用された配列を元データセットから抽出し、
1000ntを超える配列を除外する。`efold_train` 自体を学習入力にはせず、
RNAcentral由来の配列・構造ラベルは学習しない。

| 用途 | データ | 配列数 |
|---|---|---:|
| 10エポック事前学習・replay | bpRNA-1m | 45,507 |
| FT | Ribonanza (`ribo500-blast`) | 45,729 |
| FT | ヒトpri-miRNA | 1,096 |
| FT | ヒトmRNA | 1,375 |

化学データは計48,200配列、85,307プロファイル。1エポックは全48,200配列を
一度ずつ訪問し、bpRNAから12,050例をreplayする計60,250例。
replayは全学習例数の20%。同一配列のDMS/SHAPE損失は平均する。
配列単位の重みは一様で、データセット間の重み調整はしない。

3条件はIMLE-Rank、IMLE-NLL、SHAPE-fyで、それぞれ5エポック。
加えて、RNAstructure＋DMSによる付属構造が確認できるpri-miRNAとmRNAの
2,471配列を使う構造教師ありFT (`assisted`) を5エポック実行する。
この条件は1エポックあたり構造教師2,471例＋bpRNA replay618例で、
化学3条件と学習配列数は異なる。

Ribonanzaの付属構造はKaggle提供のEternaFold BPPを閾値0.5で二値化したもの。
RNAstructure assisted構造と区別し、`assisted` 条件では使わない。
pri-miRNA/mRNAの生成コードは `predict_structure=True` と
`predictStructure(sequence, dms=mutation_rate)` を使用している。
確認に使った固定版ノートブック・ソースは `data/efold_hf/raw/` に保存。

baselineは既存 `TrainA_only_new_parameter/TrainSetA.conf` と
同ディレクトリ `train-new.sh` の設定に合わせた。Mix/average、Zuker、
AdaBelief、lr=0.001、L2=0.01、熱力学的正則化=0.125、
構造FY perturb=0.75、batch=1、蓄積=1、seed=1234、AMPなし。
FTも同じoptimizer設定で、FT開始時にoptimizerを新規作成する。
LSTMだけnative Torch実装を使い、同じ構成・dropout確率のまま
ジョブ境界で乱数状態を復元できるようにする。

化学IMLEはperturb=0.1、nu=0.1、central difference、1 sample。
初回の `rank` はpairwise logisticであり、Mann–Whitney Rankではない。
Rankはpercentile teacher、tau=0.2、delta=0.002、teacher temperature=0.03、
RMS勾配正規化。NLLはsource/probeごとに独立した学習可能なWu分布。
SHAPE-fyはslope=2.6、intercept=-0.8。
化学損失の外側の重みはすべて1、正則化重みはすべて0.125。

2026-10-26の追加指示で、別条件 `mw_rank` を登録した。元の `rank` の結果は保持する。
MWはraw Mann–Whitney 1-AUROC、teacher正規化なし、temperature/delta/margin=0、
uniform weighting、central/RMS/nu=0.1/perturb=0.1/K=1。
共通のbpRNA epoch10からfresh AdaBeliefで5エポック、replay20%、正則化0.125。
モデル・optimizer設定・データ・サンプル順は元のFTと共通にし、各エポック後に評価する。
追加条件の短い配列・中央値・最長配列を検査する `preflight-arm` が成功したら
学習を自動投入する。固定実装の更新記録は
`protocol-revisions/20261026-mw-rank/change.json` に保存。

欠測値-1000は-999に置換し、実測負値と区別する。DMSはA/Cのみ。
公開RibonanzaデータのSHAPEにも上流処理による欠測があり、元の値を保持する。
bpRNA学習時は擬結び目、非標準塩基対、短すぎるループを含む参照から
最大塩基対数の非交差標準塩基対部分集合を作る。評価の参照構造は変更しない。

## 評価

事前学習10エポック後を共通epoch0 baselineとし、FT各条件のepoch1〜5を
すべて評価する。テスト結果でエポック選択やパラメータ調整はしない。

| テストデータ | 1000nt以下の評価配列数 |
|---|---:|
| PDB | 355 |
| viral_fragments | 40 |
| lncRNA | 3 |
| ArchiveII（著者コードの重複除去を現行公開版で再実行） | 313 |

塩基対の完全一致による配列macro precision/recall/F1、micro F1とTP/FP/FNを記録。
化学データがあるテスト配列にはBPP由来の非対合確率と実測反応性のSpearman相関も記録。BPPが非有限値または確率範囲外となる配列では相関を欠測として記録し、その配列と残りの構造評価は継続する。
予測時に化学反応性は入力しない。擬結び目を含む元の参照を使う。

**論文との相違:** 現在公開されているArchiveIIは3,370件。
著者 `archiveII_blast` ノートブックのbpRNA/Ribonanza/RNAstralignとの完全一致除去を
現行版で再実行すると論文の355件とは一致しない。RNAstralignはこの評価パネルの
作成にのみ使い、学習には使わない。lncRNA/PDBの公開スナップショットにも
論文との件数差がある。今回の結果を論文のテストスコアと直接同一視しない。
`efold_train` 採用配列を使って著者のBLAST除去を引き継ぐが、追加の相同性検索は
実施していない。全データの完全一致による学習・テスト分離は検査済み。

## 実行と確認

```bash
ROOT=qsub_logs/efold_hf/bprna100_ft5_replay20_20261024
.venv/bin/python scripts/efold_hf/workflow.py status --root "$ROOT"
```

初回投入済みのGPU検証が成功すると事前学習を自動投入し、完了後に
4条件のFTと評価を投入する。ワーカーは20時間で保存して後続ジョブへ引き継ぐ。
100例ごとに原子的にモデル、optimizer、乱数、エポック内位置を保存する。
中断時は最後に保存した状態から再開する。非有限損失・勾配は失敗として停止する。

`submissions.jsonl` はジョブID、`scheduler/` は実行ログ、
`runs/<条件>/progress.json` は進捗、`epoch-XXX-complete.json` は完了記録。
`evaluation/<条件>/epoch-XXX/` に配列別結果と集計、`results.csv` に比較表が出る。
共有の事前学習baselineは `evaluation/pretrain/epoch-010/`。

新規作成する場合のみ `prepare --root NEW_ROOT` → `submit --root NEW_ROOT` を使用。
既存rootへのprepareは拒否する。再投入は必ず既存ジョブの終了を確認してから、
固定済み `source/scripts/efold_hf/workflow.py work --root ROOT --arm ARM` を使う。

検証: `tests/test_efold_hf.py` の10テストが成功。MWの実際のloss factory・AUROC値と、
追加条件の各エポック評価の投入も検査。途中のエポックからの再開が
10エポックで終了することと、FT・評価の投入がepoch10を参照することを確認。
CPUで全5経路の実際の
forward/backward/optimizer stepと、実データでBPP評価経路を確認。
GPU検証では各条件の短い配列・中央値・最長配列でbackwardを実行し、
optimizerと乱数の保存復元を検査してから本学習に進む。
