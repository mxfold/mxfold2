# EternaFoldの同時学習比較（2026-10-22）

実験root: `qsub_logs/eternafold/eternafold-joint-20261022`。

## 手順の再監査

前回の反応性データ単独への追加学習は、EternaFold論文の同時学習と異なる。構造教師の損失を外しており、softplusによる正値化も公式手順にない。前回の合成2A3・DMS条件ではTestSetB F1が0.5701から0.2346・0.1325へ低下したが、この結果を公式の同時学習手順の性能として扱わない。完了したモデル・評価・凍結ソースは旧rootに保存し、旧RNAgym単独追加学習とその後続ジョブだけを停止した。

[論文のMethods、式14–17](https://www.nature.com/articles/s41592-022-01605-0)は、構造の対数尤度と反応性の周辺対数尤度を足して最適化する。構造のみはModel S、構造＋反応性はModel SCに対応する。公表EternaFoldのSCRRにはさらにriboswitchの2タスクがあるため、本比較はModel SCに相当するデータ構成である。化学尤度の係数は論文の `w_CM=0.5` を使う。

[公式データリスト作成スクリプト](https://github.com/WaymentSteeleLab/EternaFold/blob/87b9aac55cee14fd562049d08f7b92d3131f10ce/datalists/generate_datalist.sh)も、構造データと反応性データを同じ入力リストに列挙する。[公式README](https://github.com/WaymentSteeleLab/EternaFold/blob/87b9aac55cee14fd562049d08f7b92d3131f10ce/README.md#training)は `train --examplefile`、`--hyperparam_data`、`--regularize`、500/1000反復の例と、反応性の `1e-5` cutoffを説明する。`--initweights` 自体は公式に存在するが、反応性のみへの追加学習を論文の標準手順とはしていない。

## 今回実行する条件

全条件で監査済みTrainSetAの2,713構造を学習全体で保持する。元の3,166配列から453件を除外した監査・ペアラベル・採否をそのまま再利用する。構造のラベルは修正しない。

| 条件 | 構造データ | 反応性データ | 初期値 | 資源 |
|---|---|---|---|---|
| pretrain（対照） | TrainSetA | なし | ゼロ | 完了済みモデルを再利用 |
| joint_synthetic_2A3 | TrainSetA | TrainSetB合成2A3、1,094配列 | 全パラメータゼロ | cpu_40、MPI 32 |
| joint_synthetic_DMS | TrainSetA | TrainSetB合成DMS、1,094配列 | 全パラメータゼロ | cpu_40、MPI 32 |
| joint_rnagym_all | TrainSetA | RNAgym公式train全体、117,773配列・198,411観測 | 全パラメータゼロ | cpu_160、MPI 128 |

対照モデルは同じ構造入力・同じ目的関数で49反復時点にnative停止条件へ到達しており、反復上限100から500への変更はその停止点に影響しないため再学習しない。新しい同時学習3条件には、対照や公開モデルの学習済みパラメータを初期値として使わない。初期値は `data/zero.params` とSHA256で固定する。

正の反応性値を変換せずに使い、`1e-5` 以下・ゼロ・負値・非有限値・欠測sentinelを欠測値 `-1` とする。DMSはA/C、CMCTはG/Uに限定する。以前のsoftplusで保持していた負値・ゼロは、この公式cutoffに合わせた比較では化学尤度に寄与しない。除外塩基数と正の観測がないprofile数を `data/*.observations.tsv`、`plan.json` に残す。profile自体は全件保持し、観測がないprofileの尤度項は0となる。

目的関数は以下に事前固定する。

```text
mean_TrainSetA(structural NLL)
  + 0.5 * hierarchical_mean_chemical(marginal Gamma NLL)
  + ||w||^2 / (2 * 2713)
```

構造入力・化学入力はそれぞれexample weightの和を1にする。化学データは配列平均、その内側でprobe平均→condition平均→replicate平均とする。これはデータ数の大きな差を調整する今回の設計であり、論文の等重みファイル和をそのまま再現したものではない。nativeのGamma尤度・構造尤度を使い、lossの係数・損失内の平均化・初期値・最大500 iterationをtestによらず固定した。論文ではholdoutで係数を探索しているが、今回はその探索を行わない。500 iterationはepochではない。

## 実装と検証

元と同じ公式commit `87b9aac55cee14fd562049d08f7b92d3131f10ce` を使用する。上流ソースとの照合記録は旧rootの `upstream_provenance.json`、今回の全差分は新rootの `adapter.patch` にある。旧実験で数値検証した負の化学NLL、現在配列のevidence更新、helix cacheの修正、17桁出力、packed入力、L-BFGSの完全な中断再開を引き継ぐ。

今回は混在入力のうち構造ラベル付きの例だけparsabilityを検査するよう入力処理を拡張した。化学データには構造制約がないため常に空の構造が存在する。構造ラベルが不適合なら従来どおり失敗させる。計算する尤度・勾配の式は変更しない。

検証は、既存6項目に加えて、正の値が変換されないこと、混在入力の目的関数・勾配が構造と化学の和と一致すること、混在入力で不適合な構造を確実に拒否することを確認する。MPIでも混在入力を用いて学習する。

各学習ジョブは23:50:00、20時間を目安に採択済み反復の境界で完全なoptimizer状態を保存する。合成条件は各4区間、RNAgymは最大12区間を依存投入する。学習がnative停止条件または500反復に到達した後の区間は学習せず終了する。最後の区間でも未完了なら評価を失敗させ、未完了モデルを最終結果として使わない。

## 評価と操作

評価は旧実験と同じ入力と集計を使い、配列のみでMEA gamma=6およびBPPを出す。TestSetB 430配列のPPV/SEN/F1、RNAgymの同じtest panelに対するprobe/base別・cluster-macro Spearmanを記録する。TrainSetBの構造スコアを汎化性能には用いない。

投入IDと依存関係は `submissions.jsonl`、入力・コード・binaryのhashと比較設計は `plan.json`、進捗は `runs/<condition>/stderr.log` に記録する。完了後は `evaluation/<condition>/metrics.json`、`summary.tsv`、`results.json` に保存する。旧結果との違いには同時学習、公式cutoffへの変更、反復上限の変更が含まれるため、変化を同時学習だけの効果とは断定しない。

```bash
EF_TEST_BINARY="$PWD/qsub_logs/eternafold/eternafold-joint-20261022/bin/contrafold" OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/test_eternafold_joint.py
qstat -u up04167
```

`submit_joint.py` は重複投入を拒否する。投入済みrootで再実行しない。

## 投入と開始確認（2026-10-22 17:38 JST）

26ジョブを依存投入した。MPI検証9101466と前処理9101467はともに終了コード0。MPI 4プロセスと逐次版の3反復後の最大パラメータ差は `1.2434497875801753e-14`。

| 条件 | 最初のjob ID | 確認時の状態 |
|---|---:|---|
| joint_synthetic_2A3 | 9101468 | cpu_40、MPI 32で実行中 |
| joint_synthetic_DMS | 9101473 | cpu_40、MPI 32で実行中 |
| joint_rnagym_all | 9101478 | cpu_160、MPI 128を要求して資源待ち |
| 最終集計 | 9101492 | 全評価の完了待ち |

再生成で使用する正の観測塩基数は合成2A3が51,856、合成DMSが29,843、RNAgymが13,901,480。cutoffで除外した非正値はそれぞれ15,556、4,394、3,265,423。全profileに少なくとも1つ正の観測があり、配列数・profile数は維持されている。これらは投入時点の状態であり、最終的な性能改善は未確認。

## 進捗（2026-10-23 00:18 JST）

合成2条件の学習と評価ジョブはすべて `exit_status=0`。TestSetB F1とRNAgym評価cluster-macro Spearmanは次のとおり。

| 条件 | TestSetB F1 | RNAgym Spearman |
|---|---:|---:|
| TrainSetAのみ（対照） | 0.5701 | 0.4809 |
| 同時学習・合成2A3 | 0.5746 | 0.4847 |
| 同時学習・合成DMS | 0.5724 | 0.4834 |

前回の単独追加学習に比べて大幅に落ちる結果は再現せず、両指標とも対照を少し上回る。ただし、同時学習への変更に加え、反応性変換を公式cutoffに改め、iteration上限も増やしたため、改善をどの変更の効果とも単独では結論づけない。

RNAgym条件は9101478がcpu_160・MPI 128で実行中。2026-10-23 00:17 JSTのGrid Engine表示では経過3時間44分、RSS 274.6 GiB、GPU未使用。ログで確認できる最後の進捗はnative iteration 7で、初期objective 128.52から47.44まで低下している。残り11学習区間、評価9101491、集計9101492は依存待ちである。RNAgym条件の評価値と最終集計はまだない。

## 進捗・速報（2026-10-22 22:59 JST）

前処理とMPI事前検証（9101466–9101467）は終了コード0。合成2A3（9101468–9101472）と合成DMS（9101473–9101477）は学習・評価とも完了した。TestSetB全430件のF1は対照0.57009、2A3 0.57457、DMS 0.57241。RNAgym化学テストのcluster-macro Spearmanはそれぞれ0.48089、0.48468、0.48336で、合成条件の上昇幅は小さい。

TestSetBと合成TrainSetB配列をMMseqs2で40% identity・両方向80% coverageの条件で照合したところ、430件中125件にヒット（867アラインメント）、305件はヒットなしだった。したがって全430件のF1は独立汎化性能として読めない。ヒットなし305件に限定した平均F1は対照0.55922、2A3 0.56156、DMS 0.55919で、2A3の差は+0.00234、DMSはほぼ差がない。この閾値検索だけでは配列ファミリーの独立性を保証しない。アラインメント、対象ID、集計値は `qsub_logs/eternafold/eternafold-joint-20261022/data/trainsetb_testsetb_overlap_audit.json` と同ディレクトリ内のTSVに保存した。

RNAgym全体の同時学習（9101478）は、22:57 JSTの最後の取得可能なscheduler状態でcpu_160・MPI 128プロセス実行中。22:44 JST時点でnative最適化のiteration 4まで進み、完了マーカーはまだない。残りの依存セグメント（9101479–9101487、9101489–9101490）、RNAgym評価（9101491）、最終集計（9101492）はその完了待ち。今回の環境ではqstat/qacctが `getgrgid(2000) failed` で失敗するため、それ以降のscheduler状態は再取得できていない。
