from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

from mxfold2.fold.fold import AbstractFold


def compute_windowed_bpp(
    fold_model: AbstractFold,
    seq: str,
    window_size: int,
    window_step: int,
    max_internal_length: int = 30,
    max_helix_length: int = 30,
) -> list[list[tuple[int, float]]]:
    """Compute BPP in windows and merge overlapping results.

    Args:
        fold_model: The AbstractFold model instance.
        seq: Full RNA sequence string.
        window_size: Size of each window in nucleotides.
        window_step: Step size between consecutive windows.
        max_internal_length: Maximum internal loop length for DP.
        max_helix_length: Maximum helix length for DP.

    Returns:
        Merged BPP in the same format as non-windowed BPP:
        bpp[i] = [(j, prob), ...] where i and j are 1-indexed.
        bpp[0] is empty.
    """
    seq_len = len(seq)

    if not hasattr(fold_model.fold_wrapper, "compute_basepairing_probabilities"):
        raise RuntimeError(
            f"The fold wrapper {type(fold_model.fold_wrapper).__name__} does not support "
            f"BPP computation. Try using --fold Zuker or a non-Mixed model."
        )

    # Generate window boundaries
    windows: list[tuple[int, int]] = []
    start = 0
    while start < seq_len:
        end = min(start + window_size, seq_len)
        windows.append((start, end))
        if end == seq_len:
            break
        start += window_step

    logging.info(
        f"Windowed BPP: seq_len={seq_len}, window_size={window_size}, "
        f"window_step={window_step}, num_windows={len(windows)}"
    )

    # Compute BPP for each window
    window_results: list[tuple[int, int, list]] = []
    for win_idx, (start, end) in enumerate(windows):
        subseq = seq[start:end]
        logging.info(
            f"  Window {win_idx + 1}/{len(windows)}: "
            f"positions {start + 1}-{end} (length {end - start})"
        )

        with torch.no_grad():
            param = fold_model.make_param([subseq])
            param_on_cpu = fold_model.make_param_on_cpu(param[0])
            _pf, bpp = fold_model.fold_wrapper.compute_basepairing_probabilities(
                subseq,
                param_on_cpu,
                max_internal_length=max_internal_length
                if max_internal_length is not None
                else len(subseq),
                max_helix_length=max_helix_length,
                allowed_pairs=fold_model.allowed_pairs,
                constraint=None,
                reference=None,
                paired_position_scores=None,
                loss_pos_paired=0.0,
                loss_neg_paired=0.0,
                loss_pos_unpaired=0.0,
                loss_neg_unpaired=0.0,
                pairing_rules=fold_model.pairing_rules,
            )

        window_results.append((start, end, bpp))

    return merge_window_bpps(window_results, seq_len)


def merge_window_bpps(
    window_results: list[tuple[int, int, list]],
    seq_len: int,
) -> list[list[tuple[int, float]]]:
    """Merge BPP results from multiple windows by averaging overlapping pairs.

    Args:
        window_results: List of (start, end, bpp) where start/end are 0-based
            and bpp is 1-indexed sparse BPP from compute_basepairing_probabilities.
        seq_len: Length of the full sequence.

    Returns:
        Merged BPP: bpp[i] = [(j, prob), ...] where i, j are 1-indexed.
        bpp[0] is empty.
    """
    # Accumulate probabilities for each pair
    pair_accum: dict[tuple[int, int], list[float]] = defaultdict(list)

    for start, _end, bpp in window_results:
        # bpp is 1-indexed: bpp[local_i] = [(local_j, prob), ...]
        for local_i in range(1, len(bpp)):
            global_i = start + local_i  # 1-based global position
            for local_j, prob in bpp[local_i]:
                global_j = start + local_j  # 1-based global position
                pair_accum[(global_i, global_j)].append(prob)

    # Build merged sparse BPP (1-indexed, position 0 is empty)
    merged: list[list[tuple[int, float]]] = [[] for _ in range(seq_len + 1)]
    for (i, j), probs in pair_accum.items():
        avg_prob = sum(probs) / len(probs)
        merged[i].append((j, avg_prob))

    # Sort each position's pairs by partner position
    for i in range(len(merged)):
        merged[i].sort(key=lambda x: x[0])

    return merged


def centroid_from_bpp(
    bpp: list[list[tuple[int, float]]],
    seq_len: int,
    threshold: float = 0.5,
) -> tuple[str, list[int]]:
    """Generate centroid structure from BPP matrix.

    Base pairs with probability > threshold are adopted.

    Args:
        bpp: Sparse BPP matrix (1-indexed). bpp[i] = [(j, prob), ...].
        seq_len: Length of the sequence.
        threshold: Probability threshold for including a base pair.

    Returns:
        Tuple of (dot-bracket string, bp list).
        bp is 1-indexed: bp[i] = partner of position i (0 = unpaired).
    """
    bp = [0] * (seq_len + 1)  # 1-indexed
    for i in range(1, len(bpp)):
        for j, prob in bpp[i]:
            if prob > threshold and i < j and bp[i] == 0 and bp[j] == 0:
                bp[i] = j
                bp[j] = i

    pred = []
    for i in range(1, seq_len + 1):
        if bp[i] == 0:
            pred.append(".")
        elif i < bp[i]:
            pred.append("(")
        else:
            pred.append(")")
    return "".join(pred), bp
