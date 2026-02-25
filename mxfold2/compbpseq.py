from __future__ import annotations

import math
import re
from typing import Optional

import torch

from mxfold2.nucleosides import is_modified_base, get_modified_positions

MODIFIED_BASE_ALIASES: dict[str, str] = {
    "m6A": "Ж",
    "m5C": "?",
    "psi": "P",
    "pseudouridine": "P",
    "I": "I",
    "inosine": "I",
}

# Try to import C++ implementation for faster compare_bpseq
try:
    from mxfold2 import interface as _cpp

    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


def read_bpseq(
    file: str,
) -> tuple[str, list[int], Optional[str], Optional[float], Optional[float]]:
    with open(file) as f:
        p = [0]
        s = [""]
        name = sc = t = None
        for l in f:
            if l.startswith("#"):
                m = re.search(r"^# (.*) \(s=([\d.]+), ([\d.]+)s\)", l)
                if m:
                    name, sc, t = m[1], float(m[2]), float(m[3])

            else:
                idx, c, pair = l.rstrip("\n").split()
                s.append(c)
                p.append(int(pair))
    seq = "".join(s)
    return (seq, p, name, sc, t)


def read_pdb(file: str) -> list[tuple[int, int]]:
    p = []
    with open(file) as f:
        for l in f:
            l = l.rstrip("\n").split()
            if len(l) == 2 and l[0].isdecimal() and l[1].isdecimal():
                p.append((int(l[0]), int(l[1])))
    return p


def read_list_file(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def compare_bpseq_files(
    ref_path: str,
    pred_path: str,
    modified_only: bool = False,
    use_pdb: bool = False,
    modified_types: set[str] | None = None,
) -> (
    tuple[
        str | None,
        int,
        float | None,
        float | None,
        int,
        int,
        int,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
    ]
    | None
):
    if use_pdb:
        ref = read_pdb(ref_path)
        bases = None
    else:
        seq, ref, _, _, _ = read_bpseq(ref_path)
        bases = list(seq)
    seq, pred, name, sc, t = read_bpseq(pred_path)
    tp, tn, fp, fn = compare_bpseq(
        ref,
        pred,
        bases=bases,
        modified_only=modified_only,
        modified_types=modified_types,
    )
    if tp == 0 and tn == 0 and fp == 0 and fn == 0:
        return None
    sen, ppv, fval, mcc, spec, acc = accuracy(tp, tn, fp, fn)
    return (name, len(seq), t, sc, tp, tn, fp, fn, sen, ppv, fval, mcc, spec, acc)


def compare_bpseq(
    ref,
    pred,
    bases: list[str] | None = None,
    modified_only: bool = False,
    modified_types: set[str] | None = None,
) -> tuple[int, int, int, int]:
    modified_positions: set[int] | None = None
    if modified_only and bases is not None:
        seq_str = "".join(bases)
        modified_positions = get_modified_positions(seq_str, types=modified_types)
        if modified_types is not None and len(modified_positions) == 0:
            return (0, 0, 0, 0)

    if (len(ref) > 0 and isinstance(ref[0], list)) or (
        isinstance(ref, torch.Tensor) and ref.ndim == 2
    ):
        L = len(pred) - 1
        if isinstance(ref, torch.Tensor):
            ref_pairs = ref.tolist()
        else:
            ref_pairs = ref
        if _HAS_CPP and modified_positions is None:
            return _cpp.compare_bpseq_pairs(ref_pairs, pred, L)
        else:
            return _compare_bpseq_pairs_python(ref_pairs, pred, L, modified_positions)

    else:
        L = len(ref) - 1
        if isinstance(ref, torch.Tensor):
            ref = ref.tolist()
        if isinstance(pred, torch.Tensor):
            pred = pred.tolist()
        if _HAS_CPP and modified_positions is None:
            return _cpp.compare_bpseq_array(ref, pred)
        else:
            return _compare_bpseq_array_python(ref, pred, L, modified_positions)


def _compare_bpseq_pairs_python(
    ref, pred, L: int, modified_positions: set[int] | None = None
) -> tuple[int, int, int, int]:
    if isinstance(ref, torch.Tensor):
        ref = ref.tolist()
    if modified_positions is not None:
        ref = {
            (min(i, j), max(i, j))
            for i, j in ref
            if i in modified_positions or j in modified_positions
        }
        pred = {
            (i, j)
            for i, j in enumerate(pred)
            if i < j and (i in modified_positions or j in modified_positions)
        }
    else:
        ref = {(min(i, j), max(i, j)) for i, j in ref}
        pred = {(i, j) for i, j in enumerate(pred) if i < j}
    tp = len(ref & pred)
    fp = len(pred - ref)
    fn = len(ref - pred)
    if modified_positions is not None:
        n_modified = len(modified_positions)
        total_pairs = n_modified * (n_modified - 1) // 2 + n_modified * (L - n_modified)
    else:
        total_pairs = L * (L - 1) // 2
    tn = total_pairs - tp - fp - fn
    return (tp, tn, fp, fn)


def _compare_bpseq_array_python(
    ref, pred, L: int, modified_positions: set[int] | None = None
) -> tuple[int, int, int, int]:
    assert len(ref) == len(pred)
    if modified_positions is not None:
        ref_set = {
            (i, ref[i])
            for i in range(1, L + 1)
            if ref[i] > 0
            and i < ref[i]
            and (i in modified_positions or ref[i] in modified_positions)
        }
        pred_set = {
            (i, pred[i])
            for i in range(1, L + 1)
            if pred[i] > 0
            and i < pred[i]
            and (i in modified_positions or pred[i] in modified_positions)
        }
        tp = len(ref_set & pred_set)
        fp = len(pred_set - ref_set)
        fn = len(ref_set - pred_set)
    else:
        tp = fp = fn = 0
        for i, (j1, j2) in enumerate(zip(ref, pred)):
            if j1 > 0 and i < j1:
                if j1 == j2:
                    tp += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            elif j2 > 0 and i < j2:
                fp += 1
    if modified_positions is not None:
        n_modified = len(modified_positions)
        total_pairs = n_modified * (n_modified - 1) // 2 + n_modified * (L - n_modified)
    else:
        total_pairs = L * (L - 1) // 2
    tn = total_pairs - tp - fp - fn
    return (tp, tn, fp, fn)


def accuracy(
    tp: int, tn: int, fp: int, fn: int
) -> tuple[float, float, float, float, float, float]:
    # 特殊ケース1: 正解も予測もペアなし（完璧な陰性予測）
    if tp == 0 and fn == 0 and fp == 0 and tn > 0:
        return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)  # sen, ppv, fval, mcc, spec, acc

    # 通常の計算
    sen = tp / (tp + fn) if tp + fn > 0.0 else 0.0
    ppv = tp / (tp + fp) if tp + fp > 0.0 else 0.0
    fval = 2 * sen * ppv / (sen + ppv) if sen + ppv > 0.0 else 0.0

    # MCC計算（修正版）
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator > 0:
        mcc = ((tp * tn) - (fp * fn)) / math.sqrt(denominator)
    else:
        # 特殊ケース: すべて陰性で正しく予測
        if tp == 0 and fn == 0 and fp == 0 and tn > 0:
            mcc = 1.0
        # 特殊ケース: すべて陰性が正解だが誤検出あり
        elif tp == 0 and fn == 0 and fp > 0:
            mcc = -1.0
        else:
            mcc = 0.0

    # Specificity（特異度）: 「ペアを形成しない」と正しく予測した割合
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0

    # Accuracy（正解率）: 全体の正解率
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0

    return (sen, ppv, fval, mcc, specificity, acc)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="calculate SEN, PPV, F, MCC, Specificity, Accuracy for the predicted RNA secondary structure",
        add_help=True,
    )
    parser.add_argument("ref", type=str, help="BPSEQ file or list of BPSEQ files")
    parser.add_argument("pred", type=str, help="BPSEQ file or list of BPSEQ files")
    parser.add_argument("--pdb", action="store_true", help="use pdb labels for ref")
    parser.add_argument(
        "--modified-only",
        action="store_true",
        help="calculate scores only for base pairs involving modified bases",
    )
    parser.add_argument(
        "--modified-types",
        type=str,
        help="calculate scores only for specific modified base types (comma-separated: m6A,m5C,psi,I). Implies --modified-only",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="treat ref and pred as list files containing multiple BPSEQ paths",
    )
    args = parser.parse_args()

    modified_types: set[str] | None = None
    if args.modified_types:
        modified_types = set()
        for t in args.modified_types.split(","):
            t = t.strip()
            if t in MODIFIED_BASE_ALIASES:
                modified_types.add(MODIFIED_BASE_ALIASES[t])
            else:
                modified_types.add(t)
        args.modified_only = True

    if args.list:
        ref_files = read_list_file(args.ref)
        pred_files = read_list_file(args.pred)
        if len(ref_files) != len(pred_files):
            raise ValueError(
                f"list length mismatch: ref={len(ref_files)}, pred={len(pred_files)}"
            )

        print(
            "name,length,time,score,tp,tn,fp,fn,sen,ppv,fval,mcc,specificity,accuracy"
        )
        all_sen: list[float] = []
        all_ppv: list[float] = []
        all_fval: list[float] = []
        all_mcc: list[float] = []
        all_spec: list[float] = []
        all_acc: list[float] = []

        for ref_path, pred_path in zip(ref_files, pred_files):
            result = compare_bpseq_files(
                ref_path,
                pred_path,
                modified_only=args.modified_only,
                use_pdb=args.pdb,
                modified_types=modified_types,
            )
            if result is None:
                continue
            name, length, t, sc, tp, tn, fp, fn, sen, ppv, fval, mcc, spec, acc = result
            all_sen.append(sen)
            all_ppv.append(ppv)
            all_fval.append(fval)
            all_mcc.append(mcc)
            all_spec.append(spec)
            all_acc.append(acc)
            print(",".join(str(v) for v in result))

        n = len(all_sen)
        if n > 0:
            print(
                f"# avg_sen={sum(all_sen) / n:.4f}, "
                f"avg_ppv={sum(all_ppv) / n:.4f}, "
                f"avg_f1={sum(all_fval) / n:.4f}, "
                f"avg_mcc={sum(all_mcc) / n:.4f}, "
                f"avg_specificity={sum(all_spec) / n:.4f}, "
                f"avg_accuracy={sum(all_acc) / n:.4f}, "
                f"n={n}"
            )
    else:
        result = compare_bpseq_files(
            args.ref,
            args.pred,
            modified_only=args.modified_only,
            use_pdb=args.pdb,
            modified_types=modified_types,
        )
        if result is not None:
            print(", ".join(str(v) for v in result))
