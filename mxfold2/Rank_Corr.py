import sys
from pathlib import Path
import argparse
import re
from typing import Optional, Tuple, List

import pandas as pd
from scipy.stats import spearmanr

def read_bpseq(file: str) -> Tuple[str, List[int], Optional[str], Optional[float], Optional[float]]:
    """
    compbpseq に合わせた bpseq 読み込み。
    戻り値: (sequence, partner_list (1-based, p[0]=0), name, score, time)
    header に "# NAME (s=..., ...s)" のような行があればパースする。
    """
    with open(file, 'r') as f:
        p = [0]
        s = ['']
        name = sc = t = None
        for l in f:
            if not l.strip():
                continue
            if l.startswith('#'):
                m = re.search(r'^#\s*(.*)\s+\(s=([\d.]+),\s*([\d.]+)s\)', l)
                if m:
                    name, sc, t = m[1], float(m[2]), float(m[3])
                continue
            parts = l.rstrip('\n').split()
            if len(parts) < 3:
                continue
            idx, c, pair = parts[0], parts[1], parts[2]
            try:
                # append sequence char and pair index
                s.append(c)
                p.append(int(pair))
            except ValueError:
                continue
    seq = ''.join(s)
    return seq, p, name, sc, t

def read_shape(p: str) -> dict:
    d = {}
    with open(p, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('/') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
                val = float(parts[2])
            except ValueError:
                continue
            d[idx] = val
    return d

def compute_rank_corr(bpseq_path: str, shape_path: str):
    """
    1 RNA 分の解析を行い、結果を返す（CLI と同じ計算）。
    戻り値: (name, n_total, n_paired, n_unpaired, rho_paired, p_paired, rho_dist, p_dist, df)
    """
    seq, partners, name, sc, t = read_bpseq(str(bpseq_path))
    sh = read_shape(str(shape_path))

    L = len(partners) - 1
    idxs = list(range(1, L + 1))
    reactivities = [sh.get(i, float('nan')) for i in idxs]
    partner_vals = [partners[i] for i in idxs]

    df = pd.DataFrame({
        'index': idxs,
        'partner': partner_vals,
        'reactivity': reactivities
    })

    # 除外: reactivity < -10
    exclude_mask = df['reactivity'] < -10
    if int(exclude_mask.sum()) > 0:
        df = df.loc[~exclude_mask].reset_index(drop=True)

    if df.empty:
        return name, 0, 0, 0, float('nan'), float('nan'), float('nan'), float('nan'), df

    df['paired'] = (df['partner'] != 0).astype(int)
    df['pair_dist'] = df.apply(lambda r: abs(r['partner'] - r['index']) if r['partner'] != 0 else float('nan'), axis=1)

    rho_paired, p_paired = spearmanr(df['reactivity'], df['paired'], nan_policy='omit')
    paired_mask = df['paired'] == 1
    if paired_mask.sum() > 1:
        rho_dist, p_dist = spearmanr(df.loc[paired_mask, 'reactivity'], df.loc[paired_mask, 'pair_dist'], nan_policy='omit')
    else:
        rho_dist = p_dist = float('nan')

    n_total = len(df)
    n_paired = int(df['paired'].sum())
    n_unpaired = n_total - n_paired

    return name, n_total, n_paired, n_unpaired, rho_paired, p_paired, rho_dist, p_dist, df

def main(bpseq_path: Path, shape_path: Path) -> int:
    # CLI 実行は既存の出力を維持するため、compute_rank_corr を利用
    name, n_total, n_paired, n_unpaired, rho_paired, p_paired, rho_dist, p_dist, df = compute_rank_corr(str(bpseq_path), str(shape_path))
    if n_total == 0:
        print("No valid positions remaining after filtering; skipping correlation.")
        return 1
    print(f"name = {name}")
    print(f"n_total = {n_total}, n_paired = {n_paired}, n_unpaired = {n_unpaired}")
    print("Spearman reactivity vs paired_flag   : rho = {:.4f}, p = {:.4g}".format(rho_paired, p_paired))
    if not (rho_dist != rho_dist):
        print("Spearman reactivity vs pair_distance : rho = {:.4f}, p = {:.4g}".format(rho_dist, p_dist))
    else:
        print("Spearman reactivity vs pair_distance : not enough paired positions to compute")

    print("\nSample (index, partner, paired, pair_dist, reactivity):")
    print(df.head(40).to_string(index=False))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate rank correlations between .bpseq and .shape reactivities.')
    parser.add_argument('bpseq', type=Path, help='BPSEQ-formatted file (reference)')
    parser.add_argument('shape', type=Path, help='SHAPE .tsv file with reactivities')
    args = parser.parse_args()
    sys.exit(main(args.bpseq, args.shape) or 0)