# EternaFoldの共通事前学習・化学データ追加学習

**2026-10-22更新:** この逐次追加学習を公式の同時学習手順と比較し、構造損失の欠落と独自のsoftplus変換を確認した。旧RNAgym単独追加学習と後続ジョブを停止し、[構造と化学の同時学習](JOINT.md)を別rootで投入した。現在の実験設計・資源・結果の読み方はそちらを参照する。以下は完了済み結果も含む旧実験の記録である。

2026-10-22に投入した実験のrootは
`/gs/bs/tga-satolab-gtex/satoken/repos/mxfold2-ryntaro/qsub_logs/eternafold/eternafold-traina-branches-20261022`。
実行中のスクリプトはroot内の `source/` に固定している。作業側のファイルを編集しても投入済みの実験には反映されない。

共通の構造事前学習を1回行い、同じcheckpointから次の3条件へ分岐する。

| 条件 | 学習データ | 配列数 | 観測数 |
|---|---|---:|---:|
| pretrain | 監査後のTrainSetA | 2,713 | 2,713構造 |
| synthetic_2A3 | TrainSetB由来の合成2A3 | 1,094 | 1,094 |
| synthetic_DMS | TrainSetB由来の合成DMS | 1,094 | 1,094 |
| rnagym_all | RNAgym公式train全体・5プローブ | 117,773 | 198,411 |

RNAgymは以前の開発用validationも含めた公式train全体を使う。今回は変換、正則化、反復上限を事前固定しており、validationやtestによる条件・checkpoint選択を行わない。各段階はnative L-BFGSの収束または最大100 iterationで終了する。iterationはMXfold2のepochとは対応しない。replayは0%。

## 構造監査と前処理

TrainSetAの3,166配列を全件調べ、EternaFoldの標準制約で扱える2,713配列を使用する。453配列を除外し、残した構造のペアラベルは変更しない。

- 非標準ペアを含む23配列、ペア距離が4未満の417配列、曖昧塩基を含む2配列はPython監査で除外。理由には重複がある。
- 残りをEternaFoldの制約付きViterbiで検査し、追加で14配列を除外した。この14配列はすべて長さ30超のinternal/bulge loopを含み、標準の `C_MAX_SINGLE_LENGTH=30` に合わない。
- `data/TrainSetA_audit.tsv` に全配列の採否、元ファイル、理由を保存し、`data/TrainSetA_accepted.lst` に実使用リストを保存する。

化学入力は観測マスクを先に作り、観測値だけに `1e-4 + softplus(r)` を適用する。有限の負値とゼロは観測として保持し、非有限値・欠測sentinel・プローブ適用外塩基は欠測 `-1` とする。DMSはA/C、CMCTはG/Uに限定する。正解構造は化学入力に含めない。

損失の重みは、配列内でreplicate平均→condition平均→probe平均、その後に配列平均となるように設定する。全観測を使用し、全体の重みの和は1。nativeの化学尤度は各配列の観測塩基のjoint likelihoodであり、IMLE-Rankと同じ損失や塩基別正規化ではない。塩基別confidence/SNRによる追加重み付けは行わない。

正則化は全条件共通で `C=1/2713`、化学損失係数は0.5。事前学習の重みの初期値はゼロ、化学分布用の5チャネルも先に登録する。公開学習済み重みやEternaBenchの追加データは使わない。

## 上流コードの固定と修正

上流はcommit `87b9aac55cee14fd562049d08f7b92d3131f10ce`。74個のソースファイルをGit blob SHA1と照合し、すべて一致した。`upstream/` と `upstream_provenance.json`、LICENSEを保存している。

`patch_upstream.py` により、以下の修正・アダプタを適用した。実際の差分は実験rootの `adapter.patch`。

1. nativeの構造特徴量・Gamma分布・inside/outside計算を用いたまま、重み付きpacked入力を追加。約20万個の個別ファイルを作らず、反応性は疎な1チャネルとして保存し、読込時に5チャネルへ展開する。
2. 化学データの負の対数尤度が負になった場合も損失・勾配を保持する。連続密度は1を超え得るため、負のNLLをゼロにしていた上流処理を化学データには適用しない。
3. 現在の配列を読んだ後に反応性の尤度を再計算する。上流の順序では、前の配列の反応性バッファが使われ得た。
4. 通常のensembleと反応性を条件にしたensembleを切り替えるとき、共有helix cacheを再計算する。
5. L-BFGSの採択済み重み・勾配・curvature vectors・scaling・反復カウンタをatomic保存し、次のジョブで同じoptimizer状態から再開する。パラメータ入出力は17桁精度。
6. 構造制約のない化学入力ではparsabilityの全件検査を省略する。上流の検査が観測数Nの配列をN回確保する負担を避ける。構造入力の監査は省略しない。

前回の少量動作確認での有限性だけでは、上記2〜4の問題は検出できなかった。今回は重み付き目的関数の一致と、構造・Gammaパラメータの勾配を有限差分で確認している。

## 計算資源と依存関係

学習はすべて **`cpu_40=1`、MPI 32プロセス**。GPU資源は要求しない。前処理・構造監査も `cpu_40=1` を使用した。評価は `cpu_16=1`、MPI 16プロセス。

各学習段階に最大6個の継続ジョブを用意し、20時間を目安に反復の境界で状態を保存する。h_rtは23:50:00。100 iteration以前に収束した場合、その段階の残りの継続ジョブは何も学習せず正常終了する。6ジョブの総時間でも終了しない場合、未完了モデルを評価せず、完了マーカーの欠落として停止する。

事前学習が完了した後、合成2A3・合成DMS・RNAgymの3系列がそれぞれ並行開始する。各系列の内部は直列。モデルSHA256で開始点が同一であることを確認し、準備・先行ジョブが失敗していた場合は後続で学習を進めない。

| 段階 | 最初のjob ID | 状態の意味 |
|---|---:|---|
| 監査・前処理 | 9098933 | 正常終了済み |
| TrainSetA事前学習 | 9098934 | 実行中。以降の継続ジョブを依存投入済み |
| 合成2A3追加学習 | 9098941 | 共通事前学習の完了待ち |
| 合成DMS追加学習 | 9098948 | 共通事前学習の完了待ち |
| RNAgym追加学習 | 9098956 | 共通事前学習の完了待ち |
| 評価後の集計 | 9098963 | 全モデルの評価完了待ち |

上表は2026-10-22の投入時点の状態。全30ジョブのID・依存関係・要求資源は `submissions.jsonl` に記録する。

## 評価

反応性を入力せず、配列のみからMEA（gamma=6）とBPPを予測する。

- 元のTestSetB 430配列でPPV/SEN/F1を計算する。
- 既存RNAgym test panel 1,024配列・1,732観測で、元の反応性に対するprobe/base別およびcluster-macro Spearmanを計算する。
- 既存の相同性screenに加え、testのprobed coreをTrainSetBに対してMMseqs2で40% identity・80%双方向coverageの条件でscreenした。追加除外は0配列だった。RNA family独立性を保証するものではない。
- testスコアで停止点や係数を選ばない。TrainSetBを合成学習に使う条件で、TrainSetBの構造スコアを汎化評価には用いない。

完了後は `evaluation/<condition>/metrics.json`、`summary.tsv`、`results.json` に結果を保存する。事前学習のSHA256と各分岐の開始SHA256が一致することも集計時に再確認する。

## 検証・操作

`tests/test_eternafold_workflow.py` の6項目が成功した。重み付きpacked入力とテキスト入力の目的関数・勾配の一致、負の化学NLLでの勾配、異なる反応性分布での構造勾配、L-BFGSの中断再開と連続実行の一致を含む。MPI版での少量学習、配列のみのBPP/BPSEQ同時出力も成功した。

```bash
.venv/bin/python tests/test_eternafold_workflow.py
qstat -u up04167
```

主要な記録は `plan.json`（入力・実装・バイナリのSHA256）、`prepared.json`（準備完了）、`runs/<condition>/stderr.log`（反復進捗）、`runs/<condition>/complete.json`（学習完了）、`native-tests.log`（数値検証）。`submit.py` は重複投入を拒否するため、投入済みrootで再実行しない。

この比較のEternaFoldはTrainSetAの表現可能な部分集合を使うため、元の3,166配列を使ったMXfold2との教師情報量の差を結果に併記する。
