# rank loss / implicit MLE 診断

## 既存ログから分かったこと

`/gs/bs/tga-satolab-gtex/yamauchi/dev_Implicit_MLE/logs` を確認した。代表的な
`Rank-Ribo-nu10-no-grads-standardize` では、学習lossが epoch 1 の 0.833992 から
epoch 51 の 0.740864 へ低下し、Ribonanza CD80-test の SHAPE と paired/unpaired の
Spearman 相関も epoch 1 の -0.337884 から epoch 25 の -0.377228 へ改善した。
Assisted 構造に対する F 値も 0.447596 から 0.614402 へ改善している。一方、真の
TestSetB 構造 F 値は epoch 1 の 0.317400、最良 epoch 16 の 0.403669、epoch 51 の
0.361358 だった。

NLL の `nll-Ribo-m03-nu10` では TestSetB F 値が epoch 1 の 0.519093 から epoch
200 の 0.406590 へ低下した。したがって、SHAPE側の目的を下げられないことだけが
問題ではなく、SHAPE由来の paired/unpaired 信号と「正しい塩基対相手」の間の識別
不能性、および Ribonanza と TestSetB の分布差が主要因である。

## 実装上の検証対象

1. traceback の pair ベクトルと ShapeDataset の target は dummy を含む L+1 長だが、
   C++ の `paired_position_scores` は L 長の0-based配列として読む。現在は L+1の勾配を
   そのまま渡すため、base 1 に dummy の勾配、base 2 に base 1 の勾配が入り、最後の
   base の勾配は使われない。
2. implicit-MLE の有限差分勾配は通常 count差を摂動幅で割るが、現在の ADwrapper は
   `pred_counts - ref_counts` をそのまま返す。このため `nu` は構造を変える閾値だけで
   なく更新量も変え、nu sweep の解釈が難しい。
3. 非正規化の rank 勾配は全位置ペアで平均されるため長い配列ほど小さくなりやすい。
   小さい nu では2回目のMAP構造が変わらず勾配がゼロ、大きい nu では過剰摂動になる。
4. rank loss でも reactivity を `[0.01, 1.0]` にclipしている。現在の CD-80 から
   1,000配列を調べると、有効値の約11.7%が0.01未満、約5.0%が1以上で、順位情報が
   tieへ潰れる。評価用 Rank_Corr はraw値を使うため学習・評価も不一致である。
5. pure Ribonanza ジョブは空の BPSEQ list を入力し、SHAPEだけで学習しているものが
   多い。rank loss は各塩基の paired/unpaired だけを評価し、塩基対相手を評価しない。

## 実験管理にも問題がある

- `Optuna/Search-rank-Ribo` の3 trialはすべて評価値が `-inf` で、探索結果として使えない。
  qsubスクリプトは存在しない `used_data/<job>/TestSetB.lst` を `--test` に渡している。
  さらに実行コードが subprocess の標準エラーをcaptureしたまま例外を握り潰すため、失敗
  理由がジョブログに残らない。
- Optunaコードは最小の「訓練loss」のepochを選んでいる。既存ログでは訓練lossと
  TestSetB Fが逆方向に動くため、この選択基準は不適切である。
- `summary.txt` のheaderには `mean` 列がないのに、mean行には先頭に `mean` がある。
  現在の `mean_line[f_idx]` は1列ずれてFではなくPPVを返す。`f_idx + 1` が必要である。
- 保存される `.conf` はネットワーク構成だけで、loss、nu、margin、seed、optimizer、
  データlistを含まない。ディレクトリ名と実パラメータが違う例もあり、例えば
  `rank-Ribo-m03-nu10/6375067` の実際の `SHAPE_NU` は1である。
- 同じ `train` ディレクトリへresume結果を追記している系列がある。今後はjob IDごとに
  checkpoint、loss、manifest、git commit、完全な引数を分離して保存する。

## 診断ジョブ

`qsub_diagnose_activation.sh` はモデルを更新せず、既定8配列について次を比較する。
1配列・2つのnuを使ったCPUスモークでも約4分かかったため、最初から32配列にせず、
activationの傾向を確認してから `MAX_SAMPLES=32` へ増やす。

- 現行の1塩基ずれ vs dummyを除いた正しいalignment
- rankのclipあり vs raw reactivity
- 非正規化 vs 配列ごとのRMS正規化
- rank loss vs Wu NLL
- `nu = 0.1, 1, 10, 100`

各条件について、2回目のMAP構造が変わった配列率、paired状態の反転数、塩基対edgeの
変更数、目的関数の変化、count差のノルム、疑似エネルギーの大きさをJSONLと集計CSVへ
保存する。

1配列だけのローカルスモークでは、rank/current・nu=1 はMAP構造が変わらずcount差も
0だったが、dummyを除いたrank/aligned・nu=1 は50 edgeが変わり、rank目的が0.323低下
した。NLL非正規化ではnu=1が変化0、nu=10で50 edge変化だった。標本数1なので結論には
使わないが、alignmentとnuが「勾配が出る/完全に0」を切り替え得ることは確認できた。

## qsub 8631640 の結果

8配列での集計は次の通りだった（`activation_rate` は2回目foldでedgeが1つでも
変わった配列の割合）。

| profile | nu=0.1 | nu=1 | nu=10 | nu=100 |
|---|---:|---:|---:|---:|
| rank/current | 0% | 12.5% | 75% | 100% |
| rank/aligned | 0% | 25% | 87.5% | 100% |
| rank/raw-aligned | 0% | 25% | 87.5% | 100% |
| rank/raw-RMS | 75% | 100% | 100% | 100% |
| NLL/current | 0% | 0% | 37.5% | 100% |
| NLL/aligned | 0% | 0% | 25% | 100% |
| NLL/RMS | 62.5% | 100% | 100% | 100% |

非正規化の勾配RMSはrankで約`0.0117--0.0226`、NLLで約`0.0058`だった。従って
現行 `nu=0.1` は摂動が小さすぎ、`nu=100` は大きすぎる。RMS正規化は `nu=0.1` でも
activationを高くできるが、これは実効摂動を約5--9倍にしているため、学習率や
finite-difference係数と分離して評価する必要がある。

raw rankとclip rankは今回の8配列では勾配RMS差が最大でも約`1e-4`で、activationと
edge変更数も一致した。clipは仕様上の学習・評価不一致ではあるが、現状の最優先修正
対象はalignmentと摂動スケールである。

```bash
mkdir -p qsub_logs/implicit_mle_activation
qsub scripts/implicit_mle/qsub_diagnose_activation.sh
```

サンプル数を増やす場合は、例えば次のように環境変数を渡す。

```bash
qsub -v MAX_SAMPLES=32 scripts/implicit_mle/qsub_diagnose_activation.sh
```

第一段階では activation rate が 20--80% になる設定を選ぶ。0%なら勾配が死に、ほぼ
100%かつ目的関数が悪化するなら摂動過大である。第二段階では alignment修正、有限差分
の `1/nu`、raw rank、RMS正規化を実装した後、同一の事前学習checkpointから小規模な
学習ablationを行う。その際は pure SHAPE と、BPSEQ replayを混ぜた条件を分け、
SHAPE相関と TestSetB F の両方でearly stoppingする。

## 実装修正

- SHAPE target/traceback の先頭dummyを除き、fold backendへ必ず配列長 `L` の
  pseudoenergyを渡す。旧ずれ実装は削除した。
- implicit-MLEのcount差を既定で `nu` で割る。旧挙動は `none` として明示的にのみ
  選べる。
- rank targetは評価と揃えてraw reactivityを既定とし、旧clipは比較用optionにした。
- `none`、配列ごとの`rms`、`max`のpseudoenergy正規化を共通helperへ集約した。
- RMS/max正規化時の実効差分幅は `nu / scale` なので、count差の係数を
  `1 / nu` ではなく `scale / nu` に修正した。これにより正規化が更新量を意図せず
  増幅する問題を除いた。
- paired/unpaired状態の反転率が指定範囲に入るまで、配列ごとに`nu`を2倍または
  1/2倍するadaptive stepを追加した。学習はbatch size 1なので、現在は各RNAごとに
  選択される。
- 同一のperturbed parameterを正負両側で再利用するcentral differenceと、独立な
  coupled perturb-and-MAP標本を平均する`K`-sample推定を追加した。
- rank/NLL/CLSの3 lossで同じalignment・正規化・有限差分設定を使う。
- BPSEQ replayの重みを`--bpseq-loss-weight`で明示的に設定できるようにした。
  `qsub_rank_replay.sh`では既存の推奨比率（SHAPE 512件 + BPSEQ 128件）をtask 1で
  再現できる。
- 旧checkpointに任意のSHAPE回帰headがない場合だけ許容し、それ以外のmissing/unexpected
  keyは従来どおりエラーにする。

新しい推定器を有効にする代表例は次の通り。`K=2`は計算量との妥協点で、`K=4`へ
増やす場合はfold回数もほぼ比例して増える。

```bash
--shape-imle-normalization rms \
--shape-imle-difference central \
--shape-imle-samples 2 \
--shape-perturb 0.1 \
--shape-nu 0.1 \
--shape-imle-adaptive-nu \
--shape-imle-flip-low 0.01 \
--shape-imle-flip-high 0.10 \
--shape-imle-adaptive-factor 2 \
--shape-imle-adaptive-steps 4 \
--bpseq-loss-weight 1
```

今回の同規模比較は、上記の組合せを7条件へ分解した
`qsub_improved_ablation.sh`で投入する。配列タスク1〜7は、none/forward、RMS/forward、
RMS/central (K=1, 2)、adaptive、adaptive+BPSEQ replay 128、NLL+adaptive+replay
の順である。

```bash
qsub -g tga-satolab-gtex scripts/implicit_mle/qsub_improved_ablation.sh
```

## qsub 8633372: 実装後の7条件比較

上記スクリプトを `qsub -g tga-satolab-gtex` で投入した。配列タスク1〜7は全て
`qacct` の `failed=0`、`exit_status=0` で終了し、各条件のTestSetB 430件と
CD80対応配列228件を回収できた。baseline（学習前）はTestSetB F=`0.66921`、
CD80 `rho_paired=-0.44852` である。Fは高いほど、`rho_paired`は負方向ほど良い。

| condition | TestSetB F e1 → e5 | `rho_paired` e1 → e5 | 判定 |
|---|---:|---:|---|
| rank / none / forward / K1 | 0.65395 → 0.64250 | -0.47318 → -0.50141 | 既存の安定点と同等 |
| rank / RMS / forward / K1 | 0.65443 → 0.63413 | -0.47610 → -0.49651 | RMS単独では改善なし |
| rank / RMS / central / K1 | 0.52072 → 0.53330 | NaN → -0.43003 | centralが不安定 |
| rank / RMS / central / K2 | 0.51785 → 0.53926 | -0.40679 → -0.43034 | K平均だけでは回復しない |
| rank / RMS / central / K2 / adaptive | 0.65220 → 0.33000 | -0.44981 → NaN | epoch 1後に崩壊（e5で50/228件がNaN） |
| rank / RMS / central / K2 / adaptive + replay128 | 0.63667 → 0.57912 | -0.42617 → -0.44736 | この組合せは悪化 |
| **NLL / RMS / central / K2 / adaptive + replay128** | **0.67487 → 0.65610** | **-0.48213 → -0.50090** | **epoch 1がbaselineを上回る** |

最良はtask 7のepoch 1で、baseline比でTestSetB Fが`+0.00566`、
`rho_paired`が`-0.03361`となった。ただしepoch 5ではFが`-0.01311`低下するため、
この設定も5 epoch固定ではなくepoch 1付近でearly stoppingする必要がある。central
rankはK=2にしても改善せず、adaptiveでflip率（概ね2〜3%）を制御してもtask 5では
epoch 5に定数予測が増えた。従って、今回の条件ではNLLのadaptive+replay128だけが
実用候補であり、rankへ同じcentral設定を移植する根拠は得られなかった。

なお、RMS/maxの有限差分係数を修正する前のqsub 8631816のRMS結果とは直接比較せず、
今回の8633372を修正版の比較として扱う。完全な行単位の集計は
`qsub_logs/implicit_mle_improved_ablation/8633372/results_summary.csv`、ジョブログは
`qsub_logs/implicit_mle_improved_ablation/8633372/` に保存した。

## qsub 8634286: NLL (implicit MLE) 16条件比較

NLLだけの要因を分離するため、SHAPE 512件、5 epoch、seed 0、TestSetB 430件、
CD80 1842配列（対応228件）を固定し、16タスクを並列投入した。全タスクが完了markerと
epoch 1/5の評価CSVまで到達し、CD80 `rho_paired`のNaNは0件だった。baselineは
TestSetB F=`0.66921`、`rho_paired=-0.44852`である。

| condition | F e1 → e5 | `rho_paired` e1 → e5 |
|---|---:|---:|
| none / forward / K1 / nu100 | 0.65923 → 0.59039 | -0.48673 → -0.49731 |
| none / forward / K1 / nu100 + replay128 | 0.66261 → 0.62944 | -0.48580 → -0.49431 |
| RMS / forward / K1 / nu0.03 | 0.67102 → 0.65169 | -0.47592 → -0.48433 |
| RMS / forward / K1 / nu0.1 | 0.66433 → 0.63258 | -0.48049 → -0.49693 |
| RMS / forward / K1 / nu0.3 | 0.66491 → 0.62649 | -0.48508 → **-0.50940** |
| RMS / forward / K1 / nu0.1 + replay128 | 0.67285 → 0.65284 | -0.48041 → -0.49257 |
| RMS / forward / K2 / nu0.1 | 0.66358 → 0.63704 | -0.48482 → -0.49975 |
| RMS / forward / K2 / nu0.1 + replay128 | 0.66975 → 0.64762 | -0.48141 → -0.50193 |
| RMS / central / K1 / nu0.1 | 0.65981 → 0.63770 | -0.47984 → -0.49892 |
| RMS / central / K2 / nu0.1 | 0.65810 → 0.64265 | -0.48456 → -0.49924 |
| RMS / central / K2 / nu0.1 + replay128 | 0.66556 → 0.64970 | -0.48257 → -0.49930 |
| RMS / central / K2 / adaptive | 0.67529 → 0.65180 | -0.48059 → -0.50097 |
| **RMS / central / K2 / adaptive + replay128** | **0.67487 → 0.65610** | **-0.48213 → -0.50090** |
| RMS / central / K2 / adaptive + replay512 | 0.67433 → 0.65254 | -0.47851 → -0.49786 |
| RMS / central / K2 / adaptive, perturb0.05 + replay128 | 0.67428 → 0.65355 | -0.47953 → -0.49803 |
| **RMS / central / K2 / adaptive, perturb0.2 + replay128** | **0.67564 → 0.66383** | -0.47553 → -0.49861 |

結果の要点は以下である。

- epoch 1のFは perturb=`0.2` が最高（0.67564、baseline比 `+0.00643`）。ただし
  `rho_paired`はadaptive + replay128（0.67487、`-0.48213`）の方が良く、Fと相関の
  Pareto点が異なる。
- adaptiveなしでは RMS/forward/nu=`0.03` が最も高いF（0.67102）。replay128を加えると
  forward K1のFがe1で+0.00852、e5で+0.02026改善した。
- K=2は単独ではFをほぼ改善しないが、central + adaptiveでは有効な更新平均として働く。
  adaptiveなしcentral K2にreplay128を加えるとF e1が+0.00746改善した。
- replayを128から512へ増やすとF・rhoとも悪化した。摂動幅0.05は基準0.1より悪く、
  0.2はFを押し上げる一方で相関を弱める。
- adaptiveのflip率は全adaptive条件で概ね`0.029--0.031`、選択νの平均は`0.128--0.145`
  だった。NLLでは定数予測やNaNは発生していない。
- train lossの最小化順とTestSetB Fの順位は一致しない。epoch 1を検証指標で選ぶべきで、
  8633372のtask 7と今回のtask 13が同じ値を再現したため、実装は再現可能である。

従って現時点の候補は、相関を重視するなら `adaptive + replay128 + perturb0.1`、TestSetB
Fを重視するなら `perturb0.2` である。最終採用前には両条件を複数seedで再確認する。
完全な集計は `qsub_logs/implicit_mle_nll_ablation/8634286/results_summary.csv`、
投入スクリプトは `qsub_nll_ablation.sh` に保存した。

各SHAPE例について、選択された`nu`とpaired-state flip率は
`<log-dir>/imle_stats.csv`へ保存される。既定値は互換性のためforward difference、
`K=1`、adaptive無効のままである。ただしRMS/maxのfinite-difference係数は常に
修正版になるため、qsub 8631816の旧RMS結果とは直接比較せず再実行する必要がある。

単体テスト14件に加え、実モデル・実SHAPE（115 nt）のforward/backwardを確認した。
rankはcentral・`K=2`・adaptive、NLLはcentral・`K=2`・adaptive、CLSはcentral・
`K=1`・adaptiveで、それぞれ1 epochをCPUで完走した。

## qsub 8631816: 5 epoch学習ablation

同じ`TrainSetAB.pth`から、CD80 SHAPE 512件、seed 0、AdamW、lr `1e-4`で5 epoch学習した。
TestSetBは430構造、CD80は1842配列をfoldし、SHAPEファイルと対応した228配列について
`rho_paired`を集計した。SHAPE値は高いほどunpairedなので、`rho_paired`は負方向が良い。

| condition | TestSetB F e1 | F e5 | delta F | rho_paired e1 | rho e5 | delta rho |
|---|---:|---:|---:|---:|---:|---:|
| rank legacy, nu=10 | 0.64255 | 0.60254 | -0.04001 | -0.43430 | -0.43413 | +0.00017 |
| rank fixed, none, nu=10 | 0.65395 | 0.64250 | **-0.01145** | -0.47318 | **-0.50141** | **-0.02823** |
| rank fixed, RMS, nu=0.03 | 0.66503 | 0.55758 | -0.10744 | -0.47030 | -0.43128 | +0.03902 |
| rank fixed, RMS, nu=0.1 | 0.65707 | 0.63458 | -0.02249 | -0.47929 | -0.49423 | -0.01494 |
| NLL legacy, nu=100 | 0.66110 | 0.62813 | -0.03297 | -0.47865 | -0.47313 | +0.00552 |
| NLL fixed, none, nu=100 | 0.65923 | 0.59039 | -0.06885 | -0.48673 | -0.49731 | -0.01058 |
| NLL fixed, RMS, nu=0.1 | 0.66648 | 0.63984 | -0.02664 | -0.47847 | -0.49259 | -0.01412 |

旧rankはtrain lossが`0.50080 -> 0.47953`へ下がってもrhoが改善せず、TestSetB Fだけが
低下する症状を再現した。alignmentと`1/nu`を直したrank/none/nu=10はrhoを0.028改善し、
F低下も約1.1 pointに抑えたため、今回の最良設定である。RMS/nu=0.03は構造変化を発火
させても`1/nu`による更新が強すぎて崩れた。NLLも修正後はrhoが改善するが、rankより
TestSetB Fとのtrade-offが大きい。

全7 taskは末尾の完了markerまで到達し、各epoch・各条件でTestSetB 430件とCD80対応
228件の結果が揃った。集計結果は
`qsub_logs/implicit_mle_fix_ablation/8631816/results_summary.csv`に保存した。先に投入した
8631737はCD80 FASTAの選択ミス（17,037配列）を検知して取消し、1842配列のguardを追加
して8631816を再投入したため、解析には8631816だけを使う。

## qsub 8632020 / 8631942: baselineとBPSEQ replay

学習前の`TrainSetAB.pth`を同じ評価系で測ったbaselineはTestSetB F=`0.66921`、
CD80 `rho_paired=-0.44852`だった。修正版rank/none/nu=10へTrainSetAのBPSEQを混ぜた
追試は次の通り。

| condition | epoch | TestSetB F | baseline比 | rho_paired | baseline比 |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 0.66921 | -- | -0.44852 | -- |
| pure SHAPE | 1 | 0.65395 | -0.01526 | -0.47318 | -0.02466 |
| pure SHAPE | 5 | 0.64250 | -0.02671 | -0.50141 | -0.05289 |
| + BPSEQ 128 | 1 | **0.67191** | **+0.00270** | **-0.48140** | **-0.03288** |
| + BPSEQ 128 | 5 | 0.63890 | -0.03030 | -0.49912 | -0.05059 |
| + BPSEQ 512 | 1 | 0.66200 | -0.00721 | -0.48011 | -0.03158 |
| + BPSEQ 512 | 5 | 0.63272 | -0.03648 | -0.49083 | -0.04231 |

BPSEQ 128件を混ぜたepoch 1だけが、baselineのTestSetB Fを維持しながらSHAPE相関を
明確に改善したPareto点になった。512件へ増やしても改善せず、epoch 5まで学習すると
どちらもFが低下する。従って現時点の推奨は、修正版rank/none/nu=10、SHAPE 512件に
BPSEQ replay 128件、1 epochであり、train loss最小化ではなくTestSetB Fと
`rho_paired`を併用してearly stoppingする。これは単一seed・小規模実験のため、最終採用
前に3 seedで再確認する。

replayの全taskも完了markerまで到達し、集計は
`qsub_logs/implicit_mle_rank_replay/8631942/results_summary.csv`に保存した。baselineは
`qsub_logs/implicit_mle_baseline/8632020`に保存した。

## qsub 8639425: TrainSetA-only checkpointからの再評価

従来のimplicit-MLE比較は`TrainSetAB.pth`（TrainSetA+Bで学習済み）を初期値にしていたため、
TrainSetA-only baselineとしては不適切だった。そこで、TrainSetA.lstのみで学習した
`/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth`と対応する`TrainSetA.conf`
を起点に、同じSHAPE 512件、TrainSetA BPSEQ replay 128件、seed 0、AdamW、5 epochで再実験
した。baselineはepoch 0、学習条件はepoch 1/5を評価した。この`TrainSetA.pth`は
`Train_A_only`の最終保存モデル（epoch 200）であり、同ディレクトリのepoch別評価では
epoch 88付近のFが最大だったため、best epoch起点の比較は別条件になる。

| condition | epoch | TestSetB F | MCC | rho_paired | baseline比(F) |
|---|---:|---:|---:|---:|---:|
| TrainSetA baseline | 0 | 0.56132 | 0.56556 | -0.41416 | -- |
| rank fixed + replay128 | 1 | 0.55933 | 0.56376 | -0.46171 | -0.00200 |
| rank fixed + replay128 | 5 | 0.54744 | 0.55313 | -0.48884 | -0.01389 |
| NLL RMS/central/K2/adaptive + replay128, perturb=0.1 | 1 | **0.56821** | **0.57173** | -0.44520 | **+0.00688** |
| NLL RMS/central/K2/adaptive + replay128, perturb=0.1 | 5 | 0.55183 | 0.55635 | **-0.49204** | -0.00950 |
| NLL RMS/central/K2/adaptive + replay128, perturb=0.2 | 1 | 0.56669 | 0.57023 | -0.44638 | +0.00536 |
| NLL RMS/central/K2/adaptive + replay128, perturb=0.2 | 5 | 0.56648 | 0.56999 | -0.48848 | **+0.00516** |

TrainSetA起点では、NLLがepoch 1で構造Fを0.005--0.007改善し、perturb=0.2はepoch 5でも
改善を維持した。一方rankはrhoを改善するがFはbaselineを上回らなかった。従って、
TrainSetA-only条件での実用候補はNLL（F重視ならperturb=0.2、相関重視ならperturb=0.1の
early stopping）であり、TrainSetAB起点で得られたF=0.67級の値はこのbaselineには一般化
しない。なお、過去のAssisted-EternaFold pseudo-labelの最大F=0.59333はこの正しい
TrainSetA baselineおよび今回のNLLを上回るが、pseudo-label側は64,086件・旧設定、今回の
direct loss側はSHAPE 512件・現行設定なので、目的関数だけの優劣とは解釈できない。

集計は`qsub_logs/implicit_mle_trainseta/8639425/results_summary.csv`、投入スクリプトは
`qsub_trainseta.sh`、集計スクリプトは`summarize_trainseta.py`に保存した。

## Assisted foldingとのTrainSetA起点比較（qsub 8643235 / 8644447 / 8649532）

過去のpseudo-label実験の起動スクリプトを確認したところ、
`Pretrain/logs/Assisted_Eternafold/5842189/script_5842189.sh`、
`Pretrain/logs/TrA-Assisted-Ribo/6368990/script_6368990.sh`、
`Pretrain/logs/PLabel-TrA/5582802/script_5582802.sh`はいずれも
`--init-param`なしで学習を開始していた。後続ジョブも各pseudo-label実験自身の
`--resume`を使っており、`Train_A_only/TrainSetA.pth`を起点にはしていない。

この起点差を除くため、`qsub_assisted_trainseta.sh`（job 8643235）では、同じ
`/gs/bs/tga-satolab-gtex/yamauchi/Train_A_only/TrainSetA.pth`を
`--init-param`で全タスクに指定し、同じ`TrainSetA.conf`、seed 0、AdamW/lr
`1e-4`、l2 `1e-4`、FY loss、5 epoch、TestSetB 430件/CD80 1842件の評価を使う。
条件は、512 pseudo＋TrainSetA 128件、CD-80 pseudo 64,086件、CD-80 pseudo
＋TrainSetA 3,166件である。pseudo-labelは既存の
`data/Ribonanza/Assisted_Eternafold`を使用し、全64,086ファイルの存在をジョブ開始時に
検査する。各タスクはcheckpointのSHA-256と条件ファイルを保存する。

集計は`qsub_logs/implicit_mle_assisted_trainseta/<JOB_ID>/results_summary.csv`に
`summarize_assisted_trainseta.py`で出力する。

このpseudo-labelは実験的なBPSEQ正解ではない。`/gs/bs/tga-satolab-gtex/yamauchi/Tools/Eternafold.sh`
が各`shape.tsv`をBPP2SEQ（負値は`-1`）へ変換し、EternaFold/CONTRAfoldを
`--evidence --numdatasources 1 --kappa 0.1 --params EternaFoldParams_PLUS_POTENTIALS.v1`
で実行し、出力されたMEA構造を`st2bpseq.py`でBPSEQ化した固定予測である。現在のファイルは
`data/Ribonanza/Assisted_Eternafold/<ID>.bpseq`に保存されている。生成スクリプトが参照した
`Ribonanza/Shape.lst`は現在の配置にはなく、現在のIMLE直接条件が読む
`data/Ribonanza/SHAPE-rnet/<ID>.shape.tsv`（RNet 2a3推論）と同じ入力だったとは保証できない。
したがって、今回の比較は同じID集合を使うが、SHAPE入力値まで同一とはみなさない。

「512件」は`data/Ribonanza/LST/CD-80.lst`の先頭512 IDであり、512塩基や512ラベル/配列ではない。
各IDについて1本のRNA全体のpseudo BPSEQを使う。全量条件は同じリストの64,086 IDである。

タスク1（512 pseudo＋TrainSetA 128件）は完了し、TestSetB F/MCCはepoch 1で
`0.58182/0.58566`、epoch 5で`0.56451/0.56860`、CD80対応228件の
`rho_paired`は`-0.45055/-0.47164`だった。TrainSetA baseline
`0.56132/0.56556`、`-0.41416`に対して、epoch 1ではFが`+0.02050`改善する。
タスク2/3（pseudo 64,086件）は、最初のjob 8643235では8時間制限内に共有領域へ
結果を書き出せないため停止した。その後、同じ起点・条件のまま`h_rt=24:00:00`へ
延長したjob 8644447を再投入した。タスク2/3とも5 epoch完了した。
スクリプトは各epochの評価を直後に保存するため、後続epochが中断してもepoch 1の比較値を
回収できる。

再投入条件（job 8644447）は以下の通りである。

| task | pseudo-label | TrainSetA replay | 起点 |
|---:|---:|---:|---|
| 2 | CD-80全64,086件 | 0件 | `Train_A_only/TrainSetA.pth` |
| 3 | CD-80全64,086件 | 全3,166件 | `Train_A_only/TrainSetA.pth` |

全タスクの起点checkpoint SHA-256は`b11af19fb3492719bdab0534af781335831d9e576987a7ad73b0cc71a34c8e82`で一致する。

task 2（pseudoのみ）はF=`0.56018/0.55760`、rho=`-0.49288/-0.49160`（e1/e5）、
task 3（pseudo＋TrainSetA全3166件）はF=`0.54677/0.51674`、rho=`-0.48652/-0.48859`であった。
全量pseudoにTrainSetAを追加してもFは回復せず、むしろ低下した。

追加のデータ量対応として、job 8649532でpseudo 512件のみ（TrainSetA replayなし）も実行した。
この条件はTestSetB F=`0.58967/0.58746`（e1/e5）、rho=`-0.44667/-0.47943`であり、
replay128を加えたjob 8644539（F=`0.58182/0.56451`）より安定して高かった。

## IMLE-rank修正版のTrainSetA起点比較（qsub 8648412）

別スレッドで修正したIMLE-rank実装（dummy位置のalignment、finite-difference係数、
rank勾配の扱い）を使い、TrainSetA起点で8条件を再実験した。全条件はSHAPE 512件、
TrainSetA BPSEQ replay 128件、seed 0、AdamW/lr `1e-4`、5 epochで揃えた。
評価はTestSetB 430件とCD80対応228件（全unpairedになる配列では相関が未定義）である。
相関集計では、全unpaired（paired数0）の配列が出すNaNを有効配列の平均から除外している。

| condition | F e1 | F e5 | rho e1 | rho e5 |
|---|---:|---:|---:|---:|
| rank none/forward/K1, nu=10 | 0.55050 | 0.46162 | -0.44106 (227) | -0.43789 (227) |
| rank RMS/forward/K1, nu=0.1 | 0.57151 | 0.50901 | -0.45442 (228) | -0.44585 (227) |
| rank RMS/forward/K1, nu=0.03 | 0.56957 | 0.56934 | -0.42012 | -0.47019 |
| rank RMS/central/K1, nu=0.1 | 0.57061 | **0.57494** | -0.44045 | **-0.48381** |
| rank RMS/central/K2, nu=0.1 | **0.57309** | 0.57269 | -0.42433 | -0.48250 |
| rank RMS/central/K2/adaptive | 0.57275 | 0.53995 | -0.42640 | -0.45185 |
| rank RMS/central/K2/adaptive + replay512 | 0.56545 | 0.53209 | -0.43420 | -0.46533 |
| rank RMS/forward/K1, clipped | 0.56918 | 0.53798 | -0.45130 | -0.48232 |

修正版rankでは、epoch 5の最良はRMS+central+K1（F=`0.57494`, rho=`-0.48381`）、
epoch 1の最良はRMS+central+K2（F=`0.57309`）だった。非正規化nu=10は不安定で、
adaptiveやreplay512は改善せず、epoch 5にFが低下した。

同一起点で再実施したNLL 16条件（後述 qsub 8648548）では、最良Fがepoch 1で`0.57837`、
epoch 5で`0.58069`となり、rankのcentral/K1（e5 `0.57494`）を上回る。一方、rankの
central/K1はNLLのreplay128追試（e5 `0.55341`）より明確に安定しており、replay有無で
挙動が変わる。なお、NLL 16条件のqsub 8634286と旧rank条件はTrainSetAB起点のため、
TrainSetA起点の比較には使用しない。

集計は`qsub_logs/implicit_mle_rank_trainseta/8648412/results_summary.csv`、投入スクリプトは
`qsub_rank_trainseta.sh`に保存した。全8 taskは`qacct`で`failed=0`、`exit_status=0`である。

## IMLE-rankのアラインメント／有限差分スケール切り分け（qsub 8650045、旧ずれ条件はアーカイブ）

rankの主条件（RMS、central、K1、`nu=0.1`、TrainSetA replay 128件）を固定し、
アラインメントまたはcount差分のスケールだけを変えた。両taskとも`failed=0`、
`exit_status=0`である。

| condition | alignment | estimator scale | F e1 | F e5 | rho e1 | rho e5 |
|---|---|---|---:|---:|---:|---:|
| corrected reference | aligned | divide_nu | 0.57061 | 0.57494 | -0.44045 | -0.48381 |
| alignment ablation (archived old-offset behavior) | old-offset | divide_nu | 0.55494 | 0.35851 | -0.42428 | -0.19496 |
| scale ablation | aligned | none | 0.57564 | 0.57752 | -0.43224 | -0.48171 |

旧ずれ挙動（L+1 gradientのdummy indexを落とさず1塩基ずらす）はepoch 5で大きく
崩れ、アラインメント修正が必要だったことをTrainSetA起点で直接確認できた。一方、
`divide_nu`を外した条件はこの1回の5 epochではFがわずかに高いが、有限差分推定器の
スケールを失うため、原理的な修正後の既定値（`divide_nu`）を変更する根拠にはしない。
旧rank task（qsub 8639425）はalignment/estimatorの明示的な記録がない世代で、旧コードの
実挙動（L+1ベクトルの先頭L要素を使用、count差分を未除算）は、ここでの
旧ずれ挙動／`none`に相当する。そのため、旧taskとの比較は外部条件（データ、seed、
optimizer等）を揃えた修正前後比較であり、単一の内部フラグだけを変えた比較ではない。
この切り分け結果は`qsub_logs/implicit_mle_rank_trainseta_alignment/8650045/results_summary.csv`
に保存した。

## IMLE-rank式のみの厳密A/B（qsub 8650177）

アラインメント（`aligned`）、RMS正規化、central差分、`divide_nu`、raw target、
`nu=0.1`、SHAPE 512件、TrainSetA replay 128件、seed 0、optimizer、学習率、5 epochを
完全に固定し、rank目的関数だけを切り替えた。`pairwise`が修正版、
`predicted_partition`が旧式（現在の予測paired/unpairedマスクで重み付け）である。
両taskは`failed=0`、`exit_status=0`である。

| rank objective | e1 F | e5 F | e1 MCC | e5 MCC | e1 rho | e5 rho | flip e1→e5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| pairwise (修正版) | 0.57061 | **0.57494** | 0.57470 | **0.57879** | -0.44045 | **-0.48381** | 0.110→0.103 |
| predicted_partition (旧式) | 0.56551 | 0.48997 | 0.56965 | 0.50019 | -0.45401 | -0.45178 | 0.058→0.039 |

修正版はepoch 5でF `+0.08497`、MCC `+0.07860`、rho `-0.03203`となり、rank式そのものの
修正による改善を確認できた。旧式は予測構造が全pairedまたは全unpairedのとき勾配が
ゼロになり得るため、flip率も低下して学習信号が弱くなる。比較用の一時実装と投入
スクリプトは削除済みで、結果と実行時スナップショットだけを
`qsub_logs/implicit_mle_rank_formula/8650177/`に保存している。

## IMLE-NLL全条件のTrainSetA起点再実験（qsub 8648548）

以前のNLL 16条件（qsub 8634286）は`TrainSetAB.pth`起点だったため、rankとの公平な比較には
使えない。そこで同じ修正版IMLEユーティリティ、`Train_A_only/TrainSetA.pth`、SHAPE 512件、
seed 0、AdamW/lr `1e-4`、TrainSetA replay 0/128/512件、5 epochでNLLを16条件すべて再実験した。
全16 taskは`qacct`で`failed=0`、`exit_status=0`である。

| condition | F e1 | F e5 | rho e1 | rho e5 |
|---|---:|---:|---:|---:|
| none/forward/K1, nu=100 | 0.56070 | 0.51075 | -0.46450 | -0.47899 |
| none/forward/K1, nu=100 + replay128 | 0.54799 | 0.51683 | -0.45966 | -0.47545 |
| RMS/forward/K1, nu=0.03 | 0.57452 | 0.57263 | -0.43966 | -0.47814 |
| RMS/forward/K1, nu=0.1 | 0.57293 | 0.56927 | -0.44571 | -0.49078 |
| RMS/forward/K1, nu=0.3 | 0.57329 | 0.55385 | -0.46457 | -0.49756 |
| RMS/forward/K1, nu=0.1 + replay128 | 0.56852 | 0.56322 | -0.45825 | -0.48802 |
| RMS/forward/K2, nu=0.1 | 0.57311 | 0.57258 | -0.45853 | **-0.49986** |
| RMS/forward/K2, nu=0.1 + replay128 | 0.57240 | 0.54832 | -0.46357 | -0.49696 |
| RMS/central/K1, nu=0.1 | **0.57837** | **0.58069** | -0.44471 | -0.49101 |
| RMS/central/K2, nu=0.1 | 0.57815 | 0.57514 | -0.45213 | -0.48853 |
| RMS/central/K2, nu=0.1 + replay128 | 0.57390 | 0.56994 | -0.45399 | -0.49104 |
| RMS/central/K2/adaptive | 0.57237 | 0.55159 | -0.43893 | -0.48507 |
| RMS/central/K2/adaptive + replay128 | 0.56821 | 0.55183 | -0.44520 | -0.49204 |
| RMS/central/K2/adaptive + replay512 | 0.55016 | 0.51735 | -0.45114 | -0.48904 |
| RMS/central/K2/adaptive, perturb=0.05 + replay128 | 0.57047 | 0.55258 | -0.44195 | -0.48995 |
| RMS/central/K2/adaptive, perturb=0.2 + replay128 | 0.56669 | 0.56648 | -0.44638 | -0.48848 |
| RMS/central/K1, nu=0.1 + replay128（データ量対応追試） | 0.57059 | 0.55341 | -0.44866 | -0.48774 |

NLLのTestSetB FはRMS+central+K1（perturb `0.1`、replayなし）が最良で、epoch 5で
`0.58069`（MCC `0.58473`）。相関だけならRMS+forward+K2（rho=`-0.49986`）だが、Fは
`0.57258`に留まる。assistedの512件＋replay128件とのデータ量対応追試
（job 8649838）はF=`0.57059/0.55341`、rho=`-0.44866/-0.48774`となり、replayは
NLLでもFを押し下げた。adaptiveはこの起点・設定ではFを改善せず、replayを増やすほど
Fが低下した。集計は`qsub_logs/implicit_mle_nll_trainseta/8648548/results_summary.csv`および
`qsub_logs/implicit_mle_nll_trainseta_matched/8649838/results_summary.csv`に保存した。

## TrainSetA起点での現時点比較

評価値はすべてTestSetB 430構造/CD80 SHAPE対応228配列、seed 0で揃えた。F/MCCは高いほど良く、
`rho_paired`はSHAPEの「高いほどunpaired」という定義のため負方向が良い。

| method / setting | e1 F | e5 F | e1 rho | e5 rho |
|---|---:|---:|---:|---:|
| TrainSetA baseline | 0.56132 | -- | -0.41416 | -- |
| IMLE-rank RMS+central K2, nu=0.1 + replay128 | 0.57309 | 0.57269 | -0.42433 | -0.48250 |
| IMLE-rank RMS+central K1, nu=0.1 + replay128 | 0.57061 | 0.57494 | -0.44045 | -0.48381 |
| IMLE-NLL RMS+central K1, nu=0.1 | 0.57837 | 0.58069 | -0.44471 | -0.49101 |
| IMLE-NLL RMS+central K1, nu=0.1 + replay128 | 0.57059 | 0.55341 | -0.44866 | -0.48774 |
| Assisted pseudo, 512のみ（replayなし） | **0.58967** | **0.58746** | -0.44667 | -0.47943 |
| Assisted pseudo, 512 + TrainSetA replay128 | 0.58182 | 0.56451 | -0.45055 | -0.47164 |
| Assisted pseudo, CD80全64,086件 | 0.56018 | 0.55760 | -0.49288 | -0.49160 |

assistedの512件のみ条件が現時点の最高Fで、5 epochでも性能が維持された。一方、pseudoを
64,086件へ増やすとFはbaseline付近まで低下し、pseudo-labelの量を増やせば単純に良くなる
わけではない。さらにTrainSetA全3166件を追加してもFは回復しなかった。過去の
Assisted-EternaFold最高F=`0.59333`等はTrainSetA checkpointを起点にしていないため、
この比較の結論には使わない。
