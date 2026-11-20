"""Packed chemical observations and deterministic, resumable coverage cycles."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


MODIFIERS = ("2A3", "DMS", "1M7", "NMIA", "CMCT")
PROBE_BASES = {"2A3": "ACGU", "DMS": "AC", "1M7": "ACGU", "NMIA": "ACGU", "CMCT": "GU"}


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def measurement_mask(sequence: str, modifier: str, values: np.ndarray) -> np.ndarray:
    if modifier not in PROBE_BASES:
        raise ValueError(f"Unsupported chemical modifier: {modifier}")
    if len(sequence) != len(values):
        raise ValueError("Chemical measurement length differs from sequence")
    bases = np.fromiter((b in PROBE_BASES[modifier] for b in sequence), dtype=bool)
    return bases & np.isfinite(values) & (values > -900)


def observation_confidence(values, errors, snr, *, copied_error: bool):
    """Keep noisy observations; reject copied/negative/nonfinite error entries.

    The error field is NOT trusted merely because it is present. Its effect is
    bounded; unsupported error entries fall back to the profile's SNR weight.
    We do not interpret a missing reactivity as zero or a negative reactivity
    as missing. Reactivities may be negative after background subtraction.
    """
    values = np.asarray(values, dtype=np.float32)
    errors = np.asarray(errors, dtype=np.float32)
    snr = float(snr)
    if not np.isfinite(snr) or snr <= 0:
        raise ValueError("SNR must be finite and positive")
    confidence = np.full(len(values), np.clip(snr / (snr + 1), .1, 1), np.float32)
    finite = np.isfinite(values) & (values > -900)
    if not copied_error:
        valid_error = finite & np.isfinite(errors) & (errors >= 0)
        # A profile-relative noise scale accommodates distinct normalizations.
        v = values[finite]
        scale = max(float(np.quantile(v, .9) - np.quantile(v, .1)), .1) if v.size else .1
        confidence[valid_error] *= np.clip(1 / (1 + (errors[valid_error] / scale) ** 2), .1, 1)
    confidence[~finite] = 0
    return confidence


class PackedRNAGym:
    """One item per sequence, containing ALL attached chemical observations.

    Float arrays are memory mapped; metadata is compact and fully retained in
    observations.parquet. No per-measurement SHAPE files are opened at startup.
    """
    def __init__(self, folder):
        self.folder = Path(folder)
        with (self.folder / 'sequences.tsv').open() as handle:
            self.sequences = list(csv.DictReader(handle, delimiter='\t'))
        self.values = np.load(self.folder / 'values.npy', mmap_mode='r')
        self.confidence = np.load(self.folder / 'confidence.npy', mmap_mode='r')
        self.offsets = np.load(self.folder / 'observation_offsets.npy', mmap_mode='r')
        self.modifiers = np.load(self.folder / 'modifiers.npy', mmap_mode='r')
        self.conditions = np.load(self.folder / 'conditions.npy', mmap_mode='r')
        self.sequence_offsets = np.load(self.folder / 'sequence_offsets.npy', mmap_mode='r')
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())

    def __len__(self):
        return len(self.sequences)

    def indices(self, split):
        if split == 'all':
            return np.arange(len(self), dtype=np.int64)
        if split not in ('train', 'validation'):
            raise ValueError(split)
        return np.array([i for i, item in enumerate(self.sequences) if item['split'] == split], dtype=np.int64)

    def __getitem__(self, index):
        item = self.sequences[index]
        observations = []
        for observation in range(int(self.sequence_offsets[index]), int(self.sequence_offsets[index + 1])):
            left, right = map(int, self.offsets[observation:observation + 2])
            modifier = MODIFIERS[int(self.modifiers[observation])]
            values = np.array(self.values[left:right])
            confidence = np.array(self.confidence[left:right])
            mask = measurement_mask(item['sequence'], modifier, values)
            observations.append({
                'observation_index': observation, 'modifier': modifier,
                'condition_id': int(self.conditions[observation]),
                'values': torch.from_numpy(values),
                'confidence': torch.from_numpy(confidence),
                'mask': torch.from_numpy(mask),
            })
        return item, observations


def coverage_order(indices, seed, cycle):
    """Every index occurs once; reconstructing an order consumes no global RNG."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(np.unique(indices)) != len(indices) or cycle < 0:
        raise ValueError("Coverage indices must be unique and cycle nonnegative")
    return np.random.default_rng(np.random.SeedSequence([seed, cycle, 7342])).permutation(indices)


def family_weights(dataset: PackedRNAGym, indices):
    counts = {}
    for index in indices:
        key = dataset.sequences[index]['cluster']
        counts[key] = counts.get(key, 0) + 1
    raw = np.array([1 / np.sqrt(counts[dataset.sequences[i]['cluster']]) for i in indices])
    raw = np.clip(raw / raw.mean(), .1, 10)
    raw /= raw.mean()
    return {int(index): float(weight) for index, weight in zip(indices, raw)}
