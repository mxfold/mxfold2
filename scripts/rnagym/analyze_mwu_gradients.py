#!/usr/bin/env python3
"""Summarize small stochastic diagnostic windows without treating zero as cosine0."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write(path, data):
    if not data:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(data[0]), delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def summarize(grouped, labels):
    result = []
    for key, data in sorted(grouped.items()):
        item = dict(zip(labels, key))
        item['n_pairs'] = len(data)
        item['nonzero_mw_fraction'] = statistics.mean(float(row['mw_norm']) > 0 for row in data)
        conditional = [float(row['weighted_mw_over_structure']) for row in data
                       if row['weighted_mw_over_structure'] != '' and float(row['mw_norm']) > 0]
        item['weighted_mw_over_structure_nonzero_median'] = statistics.median(conditional) if conditional else ''
        item['weighted_mw_over_structure_nonzero_mean'] = statistics.mean(conditional) if conditional else ''
        for field in ('fy_loss', 'thermo_loss', 'mw_loss', 'fy_norm', 'thermo_norm', 'structure_norm',
                      'mw_norm', 'weighted_mw_norm', 'weighted_mw_over_structure',
                      'cos_fy_mw', 'cos_thermo_mw', 'cos_structure_mw', 'cos_fy_thermo'):
            values = [float(row[field]) for row in data if row[field] != '']
            item[field + '_median'] = statistics.median(values) if values else ''
            if field.startswith('cos_'):
                item[field + '_n_defined'] = len(values)
                item[field + '_negative_fraction'] = statistics.mean(x < 0 for x in values) if values else ''
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / 'gradient_summary'
    output.mkdir(exist_ok=True)
    grouped = defaultdict(list)
    sources = []
    for path in sorted((root / 'runs').glob('*/gradient_diagnostics.tsv')):
        sources.append(str(path))
        for row in rows(path):
            grouped[(path.parent.name, int(row['window']), row['group'])].append(row)
    weak = summarize(grouped, ['arm', 'window', 'group'])
    write(output / 'training_windows.tsv', weak)
    probe = root / 'b_initial_gradient/gradients.tsv'
    if probe.exists():
        sources.append(str(probe))
        grouped = defaultdict(list)
        for row in rows(probe):
            grouped[(row['condition'], row['group'])].append(row)
        b = summarize(grouped, ['condition', 'group'])
        write(output / 'b_initial.tsv', b)
        for row in b:
            if row['group'] in ('all', 'encoder', 'fc_paired'):
                print(json.dumps(row, ensure_ascii=False), flush=True)
    (output / 'scope.json').write_text(json.dumps(dict(sources=sources,
        partial=not (root / 'completed.json').exists(),
        notes=['Online stochastic task pairs, not same-RNA label contradictions.',
               'Negative cosine frequency is not the cosine of mean task gradients.',
               'Zero-norm cosines excluded, not replaced by zero.',
               'Raw gradients exclude AdamW/AdaBelief preconditioning and weight decay.',
               'A direction training_windows includes both weak arms and short strong-lambda diagnostic arms.',
               'B initial gradient probe uses coefficient .003 only for norm scaling; performs no updates.']),
        indent=2) + '\n')
    print(f'Wrote {len(weak)} training-window summaries to {output}', flush=True)


if __name__ == '__main__':
    main()
