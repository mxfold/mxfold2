"""GPU diagnostic only: compare NLL and regularizer gradients without updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


def grad_norm(gradients):
    return sum(float(g.detach().double().square().sum()) for g in gradients if g is not None)**.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, nargs='+', default=[0, 1, 5])
    parser.add_argument('--source')
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'source'))
    from scripts.efold_hf.workflow import build, digest, seed_all, verify, write_json
    verify(root)
    # Select by source/probe and length alone, without test accuracy or labels.
    selected = {}
    for line in (root/'data/chemical.jsonl').open():
        item = json.loads(line)
        if args.source and item['source'] != args.source:
            continue
        for obs in item['observations']:
            key = (item['source'], obs['probe'])
            distance = abs(len(item['sequence'])-177)
            if key not in selected or distance < selected[key][0]:
                selected[key] = (distance, item, obs)
    seed_all(1234)
    model, _, _, chemical, shapes, device = build(root, 'nll', args.gpu)
    initial_shapes = [{k: v.detach().clone() for k,v in s.state_dict().items()} for s in shapes]
    model.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    report = {'script_sha256': digest(Path(__file__)), 'model_mode': 'eval; coupled Gaussian perturbations',
              'optimizer_updates': 0, 'rows': []}
    for epoch in args.epochs:
        checkpoint = root/('runs/pretrain/epoch-010.pt' if epoch == 0 else f'runs/nll/epoch-{epoch:03d}.pt')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(payload['model_state_dict'])
        for index, shape in enumerate(shapes):
            shape.load_state_dict(payload['shape_model_state_dict'][index] if epoch else initial_shapes[index])
        for sample, ((source, probe), (_, item, obs)) in enumerate(sorted(selected.items())):
            dataset_id = 2*('ribo500-blast', 'pri_miRNA', 'human_mRNA').index(source) + int(probe == 'shape')
            target = torch.tensor([-999., *obs['values']], device=device)
            for mode in ('none', 'rms'):
                chemical.normalization = mode
                gradients = []
                losses = []
                for penalty in (0., .125):
                    chemical.sl_weight = penalty
                    seed_all(20261026+sample)
                    loss = chemical([item['sequence']], [target], dataset_id=dataset_id).mean()
                    gradients.append(torch.autograd.grad(loss, parameters, allow_unused=True))
                    losses.append(float(loss.detach()))
                difference = [(b if b is not None else 0)-(a if a is not None else 0)
                              for a,b in zip(*gradients) if a is not None or b is not None]
                row = {'epoch': epoch, 'source': source, 'probe': probe, 'id': item['id'],
                       'length': len(item['sequence']), 'normalization': mode,
                       'loss': losses[0], 'regularization_loss': losses[1]-losses[0],
                       'raw_network_gradient_norm': grad_norm(gradients[0]),
                       'regularizer_network_gradient_norm': grad_norm(difference),
                       'full_network_gradient_norm': grad_norm(gradients[1]),
                       'flip_fraction': chemical.last_imle_stats['flip_fraction']}
                report['rows'].append(row)
                print(json.dumps(row), flush=True)
    filename = 'gradients.json' if args.gpu >= 0 else 'gradients-cpu.json'
    write_json(root/'audit/20261026-nll'/filename, report)


if __name__ == '__main__':
    main()
