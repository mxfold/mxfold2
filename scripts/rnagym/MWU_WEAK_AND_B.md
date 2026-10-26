# MW弱損失比較とB方向の再検討（2026-10-11）

## 今回実行するA方向の追加実験

既存のSHAPE3000・MW事前学習checkpointを初期値にし、TrainSetA全3166件で
20epochs。AdaBelief lr0.001、weight_decay0.01、FY、score_loss_weight0.125、
perturb0.75、全層更新、accumulation1という前回と同じ条件を維持する。

固定λ0.001/0.003/0.01と、λ0.01を最初の5epochs（15830updates）で線形に0まで
減衰させる条件を各seed1234/1235/1236で比較。1構造例につき独立SHAPE1例を
同じoptimizer updateに加える。λ0になった後は補助forwardを省略する。
対照の同seed baseline20とMW-lambda0も新たに実行する（計18arms）。
同一の事前学習checkpointからのFT seed比較であり、事前学習seed比較ではない。

GPU事前検証で、診断専用の補助forwardを加えたcuDNN版λ0が通常FYと不一致だった。
再検証では補助forwardなしのcuDNN版FY分解は完全一致し、native版では補助forward・
勾配診断を加えても対照とパラメータまで完全一致した。cuDNN LSTMの内部dropout状態が
Torch/Python RNGのforkでは隔離されない挙動と整合するため、LSTMのみ
native Torch実装で実行する。CNNはcuDNNのまま、アーキテクチャ・dropout確率は不変。
以前のcuDNN対照は完全に同じ条件とは見なさず、同実装の対照を再学習する。
失敗した初回GPU検証と停止記録はimle-mwu-weak-20261011に保存し、v2を別rootにする。
v2 GPU検証は2026-10-11 09:28 JSTまでに合格。弱λの診断有無も完全一致し、
減衰条件の1epoch＋resume2epochは連続2epochsと完全一致した。

別途、λ0/0.1/0.3をseed1234で2epochs診断。各弱λ条件でも初期診断を記録する。
step1/100/300/1000から各16連続オンライン例、step3166/6332の境界例について、
FYデータ項、重み付きTurnerスコア正則化、raw MWの値・勾配ノルム・cosineを
all/embedding/encoder/fc_paired/fc_unpaired/otherで計算する。
step100の勾配は99更新後のパラメータに対する100番目の更新前の値。
ゼロノルムのcosineは未定義の空欄。optimizerの前処理・weight decayは含まない。
単独のcosineではなく、窓内の分布・非ゼロ勾配率・重み付きノルム比で判断する。
これは同一RNAへの二つのラベルの矛盾ではなく、異なるRNAを用いるタスク間の競合である。

検証は前回のTrainSetB1044と独立SHAPE512。TestSetB430はTrainSetBで選択を固定後、
選択checkpointと固定epoch20を評価。RNAgym test全件は今回のパイロットには含めない。
診断が学習を変えないこと、FY分解の等価性、減衰条件の厳密resumeをGPUで事前検証する。
各ジョブgpu_h=1、訓練はepoch10/20までの二つの4hジョブに分割して厳密resume、
訓練array同時4。BPP/BPSEQ予測中間ファイルは作成しない。

## B方向：既存の証拠

初期値は指定のTrainA_only_new_parameter/TrainSetA/epoch-100。
旧rankの小予算B方向では、512配列pool、50% replay、1024抽出/short-epoch、
accumulation4、AdamW lr1e-4、weight_decay1e-4、clip1、score_loss0、
structural perturb0.1 / chemical perturb0.1、chemical係数0.5を使用した。
TrainSetBで選んだepochは3seedとも2、TestSetB macro F1は0.617055±0.001510。
baselineは0.609251。構造だけの追加学習とchemical係数0対照はepoch0が選ばれた。
同じ短期予算のSHAPE-fy replayなしは0.615988±0.001100。
これらは新MWではなく旧pairwise-logistic rank／SHAPE-fyの結果である。
RNA単位の条件付きbootstrapであり、family独立・pretrain seed独立の証明ではない。
事前指定した改善目標には届いておらず、多数の探索後の確認的な有意性も主張しない。

一方、旧full-pool replay90 rank/2A3はTestSetB F1 0.532967、SHAPE-fy/2A3は
0.539835だった。rank/2A3のログは1epochだけで103262 draws、accumulation4で
25816 optimizer updatesを記録している。小予算rankの256 updates/short-epochと
約101倍違う。replay90%でも、総更新数と正則化条件が変われば構造精度は守れない。
この比較はpool・学習率・訓練seed等も異なるため、悪化を更新数だけの因果効果とは
断定しない。しかし「replay率を上げて同じ5epochs」という設計にはしない根拠になる。

参照：qsub_logs/rnagym/rnagym-rank-improvements-20261003-01/seed_summary.csv、
同plan.json、scripts/rnagym/REVISION_V2.md。
旧full-poolの参照：qsub_logs/rnagym/rnagym-benchmark-replay-tuned-20260921-154847/
trainseta_then_rnagym/{imle_rank_shape_replay90,shape_fy_shape_replay90}/。

## B方向の推奨次段階（本書では未投入）

追加学習は未投入だが、B初期値（baseline epoch100、structural perturb0.1、score_loss0）、
同baselineのA設定、MW事前学習済み初期値のA設定を32組の同一train RNAペアで比較する
初期勾配診断を別途投入した。モデル更新0・checkpoint保存0。
目的はBでの勾配比・競合を測り、Aでのλを無条件に流用しないことである。

この診断は2026-10-11 09:32 JSTに完了。32組のtrain RNAペア、train-mode dropout、
両勾配が非ゼロの例に限定したchemical/構造勾配ノルム比の中央値（λ0.003）は、
A MW初期値0.001668（0.167%）、B baseline初期値0.027338（2.734%）。
同じλでも相対寄与が約16倍違う。Bのλ0.01はこの初期診断で約9.11%に相当する。
これはoptimizer前処理を含まないraw勾配比であり、更新量や精度改善を保証しない。
非ゼロMW勾配率はA20/32、B21/32、cosine定義可能なのはA20、B17例だった。

A MW初期値では、FYデータ項／重み付き熱力学的スコア正則化の勾配ノルム中央値が
0.5126／9.4160だった。B初期値をA設定で測った場合は0.1223／0.1762。
A初期値の正則化再適応が強いことが示唆されるが、初期32例だけで訓練中の挙動を
断定しない。今回の100/300/1000step窓で変化を追う。
FYとMWのcosine中央値はAのfc_pairedで-0.2260、encoderで+0.0212。
層により競合の兆候は違うが、有効18例の観察なので直ちにprojectionを導入しない。

結果：qsub_logs/rnagym/imle-mwu-weak-20261011-v2/gradient_summary/b_initial.tsv。

1. MWをランダム初期値で長く学習してから戻すのではなく、構造学習済みepoch100を
   直接初期値にする。全層を小さく更新し、encoder完全freezeは初期の標準設定にしない。
2. 最初は固定のSHAPE3000 poolで、構造例512＋chemical例512を必ず同じ512 updatesに
   加算するshort-epochを最大5回。replayの割合を増やしてchemical exposureが勝手に
   増える設計や、全RNAgymを1epochとして反復する設計は避ける。
   poolは検証512と分離済み。厳密なfamily-disjointとは主張しない。
3. 更新上限2560、検証はupdate0/100/300/512/1024/1536/2560、epoch0も選択候補。
   TrainSetB macro F1とheldout SHAPE AUROCを両方追い、TestSetBで停止点を選ばない。
   過去の最良点がshort-epoch2だったことは早期評価の根拠であり、今回2が最適とは決めない。
4. optimizerは小予算Bで実績があるAdamW lr1e-4、weight_decay1e-4、clip1を出発点にし、
   lr3e-5を感度比較。構造perturb0.1、score_loss0は過去Bとの比較条件として明記する。
   A方向のbaseline再現条件（AdaBelief lr0.001、score_loss0.125、perturb0.75）とは別物。
   structural-onlyを同update・同replay露出・同optimizerで必ず併走させる。
5. MW係数0.001/0.003/0.01は今回の診断を出発点とするが、A方向の適値をそのまま
   B方向へ移さない。B初期値でも勾配比を測り、chemical寄与が構造項を圧倒しないことを
   確認する。化学損失の値の大小だけで係数を決めない。
6. 持続的な負cosineがあり弱λでも検証が悪化する場合にのみ、構造勾配を保護する
   非対称projection、またはchemical勾配をencoder中心に流す比較を追加する。
   PCGrad等はまだ実装・投入しておらず、まず診断結果で必要性を判断する。

探索中のTestSetB反復利用による楽観バイアスが残る。最終設定を検証で固定した後の
評価として、TestSetBと正しいsymmetric-bpp-v2によるRNAgym full testを報告する。
それでもTestSetBを未使用の外部検証集合とは呼ばない。
