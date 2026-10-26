"""Read-only audit of the eFold NLL checkpoints and their observation models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.stats import genextreme


SOURCES = ('ribo500-blast', 'pri_miRNA', 'human_mRNA')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'source'))
    from mxfold2.fold.shape_layers import Wu
    from scripts.efold_hf.workflow import digest, write_json
    torch.set_num_threads(4)
    grouped = defaultdict(list)
    profile_sizes = defaultdict(list)
    profile_lengths = defaultdict(list)
    sequences, profiles, bad_dms = 0, 0, 0
    for line in (root/'data/chemical.jsonl').open():
        item = json.loads(line)
        sequences += 1
        for obs in item['observations']:
            profiles += 1
            index = 2*SOURCES.index(item['source']) + int(obs['probe'] == 'shape')
            raw = np.asarray(obs['values'], dtype=np.float32)
            valid = np.isfinite(raw) & (raw > -10)
            if obs['probe'] == 'dms':
                bad_dms += sum(bool(ok) and base not in 'AC' for base, ok in zip(item['sequence'], valid))
            grouped[index].append(raw[valid])
            profile_sizes[index].append(int(valid.sum()))
            profile_lengths[index].append(len(item['sequence']))
    values = {k: np.concatenate(v) for k, v in grouped.items()}
    checkpoints = [(0, None)] + [(e, root/f'runs/nll/epoch-{e:03d}.pt') for e in range(1, 6)]
    report = {'chemical_sha256': digest(root/'data/chemical.jsonl'),
              'audit_script_sha256': digest(Path(__file__)),
              'frozen_shape_layers_sha256': digest(root/'source/mxfold2/fold/shape_layers.py'),
              'sequences': sequences, 'profiles': profiles, 'observed_DMS_GU': bad_dms,
              'epochs': [], 'checks': {}}
    for epoch, checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint else None
        row = {'epoch': epoch, 'groups': []}
        if payload:
            row['checkpoint_sha256'] = digest(checkpoint)
            row['optimizer'] = {k: payload['optimizer_state_dict']['param_groups'][0][k]
                                for k in ('lr', 'weight_decay', 'step', 'betas', 'eps')}
        for index, raw in values.items():
            model = Wu()
            if payload:
                model.load_state_dict(payload['shape_model_state_dict'][index])
            # Apply exactly the same bounds as the frozen forward path.
            model(['A'], [torch.ones(1)], [torch.tensor([.2])])
            t = torch.from_numpy(raw.clip(.01, 1.))
            with torch.no_grad():
                lp = model.paired_dist.log_prob(t)
                lu = model.unpaired_dist.log_prob(t)
                ratio = lp-lu
                z = 1+model.xi/model.sigma*(t-model.mu)
                paired_floor = lp <= np.log(1e-5)+1e-6
                lower = float(model.mu-model.sigma/model.xi)
                # Each profile's dNLL/dpaired = -log(Ppaired/Punpaired)/n_observed.
                split = torch.split(-ratio, profile_sizes[index])
                rms = np.asarray([float(x.square().sum().sqrt()/(x.numel()*np.sqrt(length)))
                                  for x, length in zip(split, profile_lengths[index]) if x.numel()])
                row['groups'].append({
                    'index': index, 'source': SOURCES[index//2],
                    'probe': 'shape' if index % 2 else 'dms', 'profiles': len(profile_sizes[index]),
                    'observed_bases': len(raw), 'raw_quantiles': np.quantile(raw, [0,.01,.25,.5,.75,.99,1]).tolist(),
                    'clipped_to_lower_fraction': float(np.mean(raw <= .01)),
                    'clipped_to_upper_fraction': float(np.mean(raw >= 1.)),
                    'Wu': {k: float(v) for k,v in model.state_dict().items()},
                    'GEV_support_lower': lower, 'outside_GEV_support_fraction': float((z <= 0).float().mean()),
                    'paired_density_floor_fraction': float(paired_floor.float().mean()),
                    'supports_unpaired_fraction': float((ratio < 0).float().mean()),
                    'log_paired_over_unpaired_quantiles': torch.quantile(ratio, torch.tensor([0., .01,.25,.5,.75,.99,1.])).tolist(),
                    'none_pseudoenergy_rms_median': float(.1*np.median(rms)),
                    'rms_normalized_pseudoenergy_rms': .1,
                    'mean_all_paired_nll': float(-lp.mean()), 'mean_all_unpaired_nll': float(-lu.mean()),
                })
        if epoch:
            results = [json.loads(line) for line in (root/f'evaluation/nll/epoch-{epoch:03d}/per_sequence.jsonl').open()]
            row['evaluation'] = {'sequences': len(results),
                                 'empty_predictions': sum(not r['predicted_pairs'] for r in results),
                                 'predicted_pairs': sum(len(r['predicted_pairs']) for r in results),
                                 'tp': sum(r['tp'] for r in results)}
        report['epochs'].append(row)

    # Independent SciPy density check, plus the analytical pairedness derivative.
    errors = []
    for epoch in report['epochs']:
        for group in epoch['groups']:
            params = group['Wu']
            model = Wu(**params)
            t = torch.linspace(.01, 1., 301, dtype=torch.float64)
            expected = genextreme.logpdf(t.numpy(), c=-params['xi'], loc=params['mu'], scale=params['sigma'])
            expected = np.maximum(expected, np.log(1e-5))
            errors.append(float(np.max(np.abs(model.paired_dist.log_prob(t).detach().numpy()-expected))))
    model = Wu()
    target = torch.tensor([-999., -.2, .1, 2.])
    paired = torch.tensor([0., .2, .5, .8], requires_grad=True)
    loss = model(['AACU'], [paired], [target]).sum()
    grad, = torch.autograd.grad(loss, paired)
    clipped = target[1:].clamp(.01, 1.)
    expected = torch.cat((torch.zeros(1), (model.unpaired_dist.log_prob(clipped)-model.paired_dist.log_prob(clipped))/3))
    report['checks'] = {'scipy_GEV_max_abs_error': max(errors),
                        'NLL_pairedness_derivative_max_abs_error': float((grad-expected).abs().max()),
                        'masked_dummy_gradient': float(grad[0]),
                        'Gamma_parameter_identity': model.unpaired_dist.concentration is model.alpha and model.unpaired_dist.rate is model.beta,
                        'GEV_parameter_identity': model.paired_dist.mu is model.mu}
    assert bad_dms == 0
    assert max(errors) < 1e-4
    assert report['checks']['NLL_pairedness_derivative_max_abs_error'] < 1e-6
    output = root/'audit/20261026-nll/audit.json'
    write_json(output, report)
    print(json.dumps({'output': str(output), 'checks': report['checks'],
                      'evaluation': [e.get('evaluation') for e in report['epochs'] if e['epoch']]}, indent=2))


if __name__ == '__main__':
    main()
