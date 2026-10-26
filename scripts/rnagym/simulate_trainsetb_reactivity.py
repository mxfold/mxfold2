#!/usr/bin/env python3
"""Attach structure-dependent pseudo 2A3/DMS profiles to a BPSEQ list.

Fit only the packed official RNAgym *train* split. Wu's noisy paired GEV /
unpaired gamma model (see mxfold2/utils/sim_shape.py) provides the ordering;
an empirical quantile transform matches the measured marginal within each
probe/base. This is a simulation assumption, not a fitted conditional model
of real paired/unpaired RNAgym measurements. Missingness is independent of
structure within base, with the observed RNAgym base-specific rates rescaled
to the requested fraction of ALL nucleotides. DMS never measures G or U.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import gamma, genextreme, ks_2samp, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mxfold2.dataset import BPseqDataset
from mxfold2.rnagym import MODIFIERS, PROBE_BASES, PackedRNAGym, measurement_mask

DEFAULT_DATA = ROOT / 'qsub_logs/rnagym/rnagym-full-coverage-20261003-01/data'
DEFAULT_LIST = DEFAULT_DATA.parent / 'TrainSetB.lst'
PROBES = ('2A3', 'DMS')
SUMMARY_Q = [0, .01, .05, .25, .5, .75, .9, .95, .99, .999, 1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summary(values):
    values = np.asarray(values)
    return {'count': len(values), 'mean': float(values.mean()) if len(values) else None,
            'negative_fraction': float((values < 0).mean()) if len(values) else None,
            'quantiles': dict(zip(map(str, SUMMARY_Q), map(float, np.quantile(values, SUMMARY_Q))))
            if len(values) else {}}


def fit_reference(folder):
    data = PackedRNAGym(folder)
    parts = {p: {b: [] for b in PROBE_BASES[p]} for p in PROBES}
    counts = {p: {'profiles': 0, 'total_bases': 0, 'valid_bases': 0,
                  'base_total': dict.fromkeys('ACGU', 0), 'base_valid': dict.fromkeys('ACGU', 0)}
              for p in PROBES}
    for index in data.indices('train'):
        sequence = data.sequences[int(index)]['sequence']
        bases = np.array(list(sequence))
        left, right = map(int, data.sequence_offsets[index:index + 2])
        for observation in range(left, right):
            probe = MODIFIERS[int(data.modifiers[observation])]
            if probe not in parts:
                continue
            start, stop = map(int, data.offsets[observation:observation + 2])
            values = np.asarray(data.values[start:stop])
            mask = measurement_mask(sequence, probe, values)
            row = counts[probe]
            row['profiles'] += 1
            row['total_bases'] += len(sequence)
            row['valid_bases'] += int(mask.sum())
            for base in 'ACGU':
                bm = bases == base
                row['base_total'][base] += int(bm.sum())
                row['base_valid'][base] += int((bm & mask).sum())
                if base in parts[probe]:
                    parts[probe][base].append(values[bm & mask].copy())
    samples = {}
    for probe in PROBES:
        row = counts[probe]
        row['fraction_of_all_bases'] = row['valid_bases'] / row['total_bases']
        row['base_observation_probability'] = {
            b: row['base_valid'][b] / row['base_total'][b] if row['base_total'][b] else 0.
            for b in 'ACGU'}
        samples[probe] = {}
        for base in PROBE_BASES[probe]:
            samples[probe][base] = np.sort(np.concatenate(parts[probe][base]))
            if not len(samples[probe][base]):
                raise ValueError(f'No reference values for {probe}/{base}')
        row['distribution_by_base'] = {b: summary(v) for b, v in samples[probe].items()}
        row['distribution'] = summary(np.concatenate(list(samples[probe].values())))
    return samples, counts


def coverage_counts(bases, probe, fraction, reference_rates):
    """Allocate an exact global count; never exceed the base-specific capacity."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Coverage must be a finite fraction between 0 and 1')
    allowed = PROBE_BASES[probe]
    sizes = np.array([np.count_nonzero(bases == b) for b in allowed])
    target = round(len(bases) * fraction)
    if target > int(sizes.sum()):
        raise ValueError(f'{probe} coverage {fraction:.5f} exceeds applicable-base fraction '
                         f'{sizes.sum() / len(bases):.5f}')
    rates = np.array([reference_rates[b] for b in allowed])
    low, high = 0., 1.
    while np.dot(sizes, np.minimum(1., high * rates)) < target:
        high *= 2
        if high > 1e12:
            raise ValueError('Reference base rates cannot supply requested coverage')
    for _ in range(80):
        mid = (low + high) / 2
        if np.dot(sizes, np.minimum(1., mid * rates)) < target:
            low = mid
        else:
            high = mid
    ideal = sizes * np.minimum(1., high * rates)
    allocation = np.floor(ideal).astype(int)
    remainder = target - int(allocation.sum())
    for i in np.argsort(-(ideal - allocation)):
        if remainder and allocation[i] < sizes[i]:
            allocation[i] += 1
            remainder -= 1
    if remainder or allocation.sum() != target:
        raise AssertionError('Coverage allocation failed')
    return dict(zip(allowed, map(int, allocation)))


def wu_scores(paired, rng):
    """Same Wu parameters as sim_shape.py; retain continuous scores for ranking."""
    paired = np.asarray(paired, dtype=bool)
    scores = np.empty(len(paired))
    scores[paired] = genextreme.rvs(-.774, loc=.078, scale=.083,
                                    size=int(paired.sum()), random_state=rng)
    scores[~paired] = gamma.rvs(1.006, scale=1 / 1.404,
                              size=int((~paired).sum()), random_state=rng)
    return scores


def simulate(bases, paired, probe, fraction, samples, rates, rng):
    allocation = coverage_counts(bases, probe, fraction, rates)
    values = np.full(len(bases), -999., dtype=np.float32)
    for base, count in allocation.items():
        positions = rng.choice(np.flatnonzero(bases == base), count, replace=False)
        scores = wu_scores(paired[positions], rng)
        order = np.argsort(scores, kind='stable')
        # Stratified empirical quantiles retain tails without adding arbitrary
        # clipping or a parametric assumption to the measured RNAgym marginals.
        probabilities = (np.arange(count) + rng.random(count)) / max(1, count)
        values[positions[order]] = np.quantile(samples[base], probabilities).astype(np.float32)
    return values


def structure_auc(values, paired):
    unpaired_values, paired_values = values[~paired], values[paired]
    if not len(unpaired_values) or not len(paired_values):
        return None
    return float(mannwhitneyu(unpaired_values, paired_values).statistic
                 / (len(unpaired_values) * len(paired_values)))


def generate(args):
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite simulation: {out}')
    dataset = BPseqDataset(str(args.bpseq_list))
    if not len(dataset):
        raise ValueError('Empty BPSEQ list')
    sequences = [sequence.upper().replace('T', 'U') for _, sequence, _ in dataset]
    paired_parts = []
    for (name, _, target), sequence in zip(dataset, sequences):
        pairs = target['target'].numpy()[1:]
        if len(pairs) != len(sequence) or not set(sequence) <= set('ACGURYSWKMBDHVN'):
            raise ValueError(f'Invalid sequence/structure in {name}')
        for i, partner in enumerate(pairs, 1):
            if partner < 0 or partner > len(pairs) or partner == i:
                raise ValueError(f'Incomplete or invalid reference structure: {name}/{i}')
            if partner and pairs[partner - 1] != i:
                raise ValueError(f'Nonreciprocal base pair in {name}/{i}')
        paired_parts.append(pairs > 0)
    bases = np.array(list(''.join(sequences)))
    paired = np.concatenate(paired_parts)
    offsets = np.concatenate(([0], np.cumsum(list(map(len, sequences)))))
    samples, reference = fit_reference(args.reference)
    # Check the packed calibration inputs against their audited manifest.
    reference_manifest = json.loads((args.reference / 'manifest.json').read_text())
    reference_hashes = {}
    for name in ('values.npy', 'modifiers.npy', 'sequence_offsets.npy',
                 'observation_offsets.npy', 'sequences.tsv'):
        actual = digest(args.reference / name)
        if actual != reference_manifest['packed_files_sha256'][name]:
            raise ValueError('Reference input changed: ' + name)
        reference_hashes[name] = actual
    out.mkdir(parents=True)
    calibration = out / 'calibration'
    calibration.mkdir()
    report = {'schema': 'trainsetb-rnagym-simulation-v1', 'seed': args.seed,
              'bpseq_list': str(args.bpseq_list.resolve()),
              'bpseq_list_sha256': digest(args.bpseq_list),
              'bpseq_sha256': {str(Path(name).resolve()): digest(name) for name, _, _ in dataset},
              'reference': str(args.reference.resolve()), 'reference_split': 'train',
              'validation_used_for_fit': False, 'test_used_for_fit': False,
              'reference_input_sha256': reference_hashes,
              'sequences': len(dataset), 'total_bases': len(bases),
              'ambiguous_bases_left_missing': int((~np.isin(bases, list('ACGU'))).sum()),
              'method': 'Wu paired GEV/unpaired gamma ordering, empirical within-probe/base quantile mapping',
              'conditional_model': 'Wu simulation assumption; RNAgym structure labels are not used or inferred',
              'missingness': 'Base-stratified uniform sampling without replacement, independent of pairedness; exact global coverage',
              'coverage_denominator': 'All nucleotides, including DMS-inapplicable G/U',
              'reference_summary': reference, 'simulation': {}}
    for probe_index, probe in enumerate(PROBES):
        requested = getattr(args, 'coverage_' + probe.lower())
        if requested is None:
            requested = args.coverage
        fraction = reference[probe]['fraction_of_all_bases'] if requested is None else requested
        values = simulate(bases, paired, probe, fraction, samples[probe],
                          reference[probe]['base_observation_probability'],
                          np.random.default_rng(np.random.SeedSequence([args.seed, probe_index])))
        folder = out / probe
        folder.mkdir()
        shape_dir = folder / 'shape'
        shape_dir.mkdir()
        shape_paths, rows = [], []
        for i, ((name, _, _), sequence) in enumerate(zip(dataset, sequences)):
            shape = shape_dir / f'{i:04d}-{Path(name).stem}.shape'
            local = values[offsets[i]:offsets[i + 1]]
            shape.write_text(''.join(f'{j}\t{base}\t{float(value):.9g}\n'
                                     for j, (base, value) in enumerate(zip(sequence, local), 1)))
            shape_paths.append(str(shape))
            rows.append({'index': i, 'sequence': sequence, 'cluster': f'TrainSetB-{i}', 'split': 'train'})
        (folder / 'TrainSetB.shape.lst').write_text('\n'.join(shape_paths) + '\n')
        with (folder / 'sequences.tsv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        for name, array in {'values': values, 'confidence': (values > -900).astype(np.float32),
                            'observation_offsets': offsets, 'sequence_offsets': np.arange(len(dataset) + 1),
                            'modifiers': np.full(len(dataset), MODIFIERS.index(probe), np.int8),
                            'conditions': np.zeros(len(dataset), np.int32)}.items():
            np.save(folder / f'{name}.npy', array)
        valid = measurement_mask(''.join(sequences), probe, values)
        by_base = {}
        for base in PROBE_BASES[probe]:
            np.save(calibration / f'{probe}-{base}-quantiles.npy',
                    np.quantile(samples[probe][base], np.linspace(0, 1, 10001)))
            mask = valid & (bases == base)
            by_base[base] = {'distribution': summary(values[mask]),
                             'fraction_observed': float(mask.sum() / (bases == base).sum()),
                             'KS_against_RNAgym_train': float(ks_2samp(values[mask], samples[probe][base]).statistic)
                             if mask.any() else None,
                             'unpaired_vs_paired_AUC': structure_auc(values[mask], paired[mask])}
        generated = {'probe': probe, 'sequences': len(dataset), 'observations': len(dataset),
                     'requested_fraction_of_all_bases': fraction,
                     'observed_fraction_of_all_bases': float(valid.mean()),
                     'valid_bases': int(valid.sum()), 'total_bases': len(values),
                     'profiles_without_values': sum(not valid[a:b].any() for a, b in zip(offsets[:-1], offsets[1:])),
                     'distribution': summary(values[valid]), 'by_base': by_base}
        hashes = {p.name: digest(p) for p in folder.glob('*.npy')}
        hashes['sequences.tsv'] = digest(folder / 'sequences.tsv')
        write_json(folder / 'manifest.json', {**generated, 'packed_files_sha256': hashes,
                                              'synthetic': True, 'seed': args.seed})
        report['simulation'][probe] = generated
    write_json(out / 'simulation_report.json', report)
    print(json.dumps({'output': str(out), 'reference_coverage': {
        p: reference[p]['fraction_of_all_bases'] for p in PROBES},
        'generated': report['simulation']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--bpseq-list', type=Path, default=DEFAULT_LIST)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261020)
    parser.add_argument('--coverage', type=float, help='Fraction of ALL bases measured for both probes (0..1)')
    parser.add_argument('--coverage-2a3', type=float, help='Override 2A3 fraction; default: official RNAgym train')
    parser.add_argument('--coverage-dms', type=float, help='Override DMS fraction; cannot exceed the A/C fraction')
    generate(parser.parse_args())


if __name__ == '__main__':
    main()
