from __future__ import annotations

import os
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional
import numpy as np

import torch
#import torch.nn as nn
#import torch.nn.functional as F
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from . import interface
from .compbpseq import accuracy, compare_bpseq
from .dataset import BPseqDataset, FastaDataset, MultiTaskDataset
from .fold.fold import AbstractFold
from .common import Common

from .shape_model import build_shape_model
from .compShape import compare_shape  # 追加

class Predict(Common):
    def __init__(self):
        super(Predict, self).__init__()

    def predict(self, 
                model: AbstractFold | AveragedModel,
                data_loader: DataLoader,
                task: str,
                output_bpseq: Optional[str] = None, 
                output_bpp: Optional[str] = None, 
                result: Optional[str] = None, 
                use_constraint: bool = False,
                shape_list: Optional[list[str]] = None,
                shape_intercept: float = 0.0,
                shape_slope: float = 0.0,
                use_amp: bool = False,
                ) -> None:

        res_fn = open(result, 'w') if result is not None else None
        shape_list = [None] * len(data_loader) if shape_list is None else shape_list
        while len(shape_list) < len(data_loader):
            shape_list.append(None)

        model.eval()
        seq_processed = 0
        with torch.no_grad():
            for headers, seqs, vals in data_loader:
                start = time.time()
                if use_constraint:
                    constraint = [ tgt if tp=='BPSEQ' else None for tp, tgt in zip(vals['type'], vals['target'])]
                else:
                    constraint = None
                # Assisted_Foldingの場合はシュードエナジーが必要。それ以外はいらない    
                if task == "Assisted_Folding":
                    pseudoenergy = [
                        self.load_shape_reactivity(shape_file, shape_intercept, shape_slope)
                        if shape_file is not None else None
                        for shape_file in shape_list[seq_processed:seq_processed+len(seqs)]
                    ]
                else:
                    pseudoenergy = [None] * len(seqs)
                
                seq_processed += len(seqs)
                
                # Use autocast for mixed precision inference
                with autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    if output_bpp is None:
                        scs, preds, bps = model(seqs, constraint=constraint, pseudoenergy=pseudoenergy)
                        pfs = bpps = [None] * len(preds)

                    else:
                        scs, preds, bps, pfs, bpps = model(seqs, return_partfunc=True, constraint=constraint, pseudoenergy=pseudoenergy)
                # マルチタスクの時はシェイプもかえす
                if task == "Multitask":
                    shp_pred = model.zuker.net.predict_shape(seqs)
                    pred_shapes = shp_pred.float().cpu().numpy()

                elif task == "Implicit_MLE":
                    device = next(self.shape_model.parameters()).device
                    paired = []
                    for bp in bps:
                        p = [1 if v > 0 else 0 for v in bp]
                        p = torch.tensor(p, dtype=torch.float32, device=device)
                        paired.append(p) 
                    targets = [t.to(device) for t in vals['shape_target']]       # ← GPUに揃える

                    if self.shape_model_name in ("Wu", "Foo"):
                        nlls = self.shape_model(seqs, paired, targets)
                        metrics = {"NLL": nlls.mean().item()}
                    elif self.shape_model_name in ("MLP", "External"):

                        shape_preds = self.shape_model.predict(seqs, paired)
                        
                        metrics = compare_shape(targets[0][1:], shape_preds[0])

                    pred_shapes = [None] * len(seqs)

                else:
                    pred_shapes = [None] * len(seqs)

                elapsed_time = time.time() - start
                # for header, seq, ref, sc, pred, bp, pf, bpp, shp in zip(headers, seqs, vals['target'], scs, preds, bps, pfs, bpps, pred_shapes):
                for i, (header, seq, sc, pred, bp, pf, bpp, shp) in enumerate(
                                zip(headers, seqs, scs, preds, bps, pfs, bpps, pred_shapes)):       
                    ref = vals['target'][i]

                    if output_bpseq is None:# コマンドラインへの出力　

                        print('>'+header)
                        print(seq)
                        print(pred, f'({sc:.1f})')
                        # ドットブラケットの下にシェイプを表示
                        if task == "Multitask" and shp is not None:
                            shp_line = " ".join(f"{val:.3f}" for val in shp)
                            print(shp_line)

                    elif output_bpseq == "stdout": # bpseq出力

                        print(f'# {header} (s={sc:.1f}, {elapsed_time:.5f}s)')
                        for i in range(1, len(bp)):
                            # 4列目にシェイプを書き加える
                            if task == "Multitask" and shp is not None:
                                print(f'{i}\t{seq[i-1]}\t{bp[i]}\t{shp[i-1]:.3f}')                                
                            else:
                                print(f'{i}\t{seq[i-1]}\t{bp[i]}')

                    else:
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(output_bpseq, fn+".bpseq")
                        with open(fn, "w") as f:
                            print(f'# {header} (s={sc:.1f}, {elapsed_time:.5f}s)', file=f)
                            for i in range(1, len(bp)):
                                # print(f'{i}\t{seq[i-1]}\t{bp[i]}', file=f)
                                if task == "Multitask" and shp is not None:
                                    print(f'{i}\t{seq[i-1]}\t{bp[i]}\t{shp[i-1]:.3f}', file=f)
                                else:
                                    print(f'{i}\t{seq[i-1]}\t{bp[i]}', file=f)

                    if res_fn is not None:
                        # 最初の1回だけヘッダーを書き込む
                        if res_fn.tell() == 0:
                            names = ["name", "length", "elapsed_time", "score",
                                    "TP", "TN", "FP", "FN", "SEN", "PPV", "F", "MCC"]
                            if task == "Multitask":
                                names += ["mse", "r2", "mae"]
                            elif task == "Implicit_MLE":
                                if "R2" in metrics:  # Regression
                                    names += ["r2", "mae"]
                                elif "NLL" in metrics:  # Wu/Foo likelihood
                                    names += ["nll"]
                            res_fn.write(",".join(names) + "\n")   # ✅ ヘッダーはここでだけ書く

                        x = compare_bpseq(ref, bp)
                        struct_metrics = [header, len(seq), elapsed_time, sc.item()] + list(x) + list(accuracy(*x))

                        # --- Multitask のときだけ SHAPE 評価を追加 ---
                        if task == "Multitask" and pred_shapes is not None:
                            shp_true = vals['shape_target'][0].cpu().numpy()
                            shp_mask = vals['shape_mask'][0].cpu().numpy()
                            shp_pred = shp

                            valid = shp_mask > 0
                            if valid.sum() > 0:
                                y_true = shp_true[valid]
                                y_pred = shp_pred[valid]

                                mse = np.mean((y_pred - y_true) ** 2)
                                r2 = 1 - ((y_pred - y_true)**2).sum() / ((y_true - y_true.mean())**2).sum()
                                corr = np.corrcoef(y_pred, y_true)[0,1]
                            else:
                                mse, r2, corr = np.nan, np.nan, np.nan

                            struct_metrics += [mse, r2, corr]
                            # struct_metrics += [round(mse, 3), round(r2, 3), round(corr, 3)]
                        elif task == "Implicit_MLE":
                            if "R2" in metrics:  # Regression
                                struct_metrics += [metrics["R2"], metrics["MAE"]]
                            elif "NLL" in metrics:  # Wu/Foo
                                struct_metrics += [metrics["NLL"]]


                        res_fn.write(', '.join([str(v) for v in struct_metrics]) + "\n")
                            
                    if output_bpp is not None:
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(output_bpp, fn+".bpp")
                        with open(fn, "w") as f:
                            for i in range(1, len(bpp)):
                                print(f"{i} {seq[i-1]} ", end='', file=f)
                                for j, p in bpp[i]:
                                    print(f"{j}:{p:.3f}", end=' ', file=f)
                                print(file=f)

    def run(self, args: Namespace, conf: Optional[str] = None) -> None:
        torch.set_num_threads(args.threads)
        interface.set_num_threads(args.threads)

        # test_dataset = FastaDataset(args.input)
        # if len(test_dataset) == 0:
        #     test_dataset = BPseqDataset(args.input)

        # まずFASTAかBPSEQかを判定
        tmp = FastaDataset(args.input)
        is_fasta = len(tmp) > 0

        if args.task in ("Multitask", "Implicit_MLE"):
            if args.shape is None:
                raise ValueError(f"{args.task} requires --shape (SHAPE file list)")
            test_dataset = MultiTaskDataset(bpseq_list=args.input, shape_list=args.shape, dataset_id=0)

        elif is_fasta:
            test_dataset = tmp  # FastaDataset
        else:
            test_dataset = BPseqDataset(args.input)
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        model, _ = self.build_model(args)
        if args.param != '':
            param = Path(args.param)
            if not param.exists() and conf is not None:
                param = Path(conf).parent / param
            p = torch.load(param, map_location='cpu',weights_only=True)
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            if 'n_averaged' in p:
                model = AveragedModel(model)
            missing, unexpected = model.load_state_dict(p, strict=False)
            # Older bundled checkpoints predate the optional SHAPE regression
            # heads. They are not needed for ordinary Folding prediction, but
            # all other incompatibilities should still fail loudly.
            optional_missing = [key for key in missing
                                if args.task in ('Folding', 'Assisted_Folding')
                                and '.shape_head_' in key]
            required_missing = [key for key in missing if key not in optional_missing]
            if required_missing or unexpected:
                details = []
                if required_missing:
                    details.append(f'missing keys: {required_missing}')
                if unexpected:
                    details.append(f'unexpected keys: {list(unexpected)}')
                raise RuntimeError('Incompatible model checkpoint (' + '; '.join(details) + ')')
        
            # --- shape model をロード ---
            if args.task == "Implicit_MLE":
                self.shape_model = build_shape_model(args)
                self.shape_model_name = args.shape_model  # ← ここで保持
                if isinstance(p, dict) and 'shape_model_state_dict' in p:
                    self.shape_model.load_state_dict(p['shape_model_state_dict'][0])

        if args.gpu >= 0:
            model.to(torch.device("cuda", args.gpu))

            if args.task == "Implicit_MLE":
                self.shape_model.to(torch.device("cuda", args.gpu))

        shape_list = None
        if args.shape is not None: 
            shape_list = []
            with open(args.shape) as f:
                for l in f:
                    l = l.rstrip('\n').split()
                    shape_list.append(l[0])
        elif args.shape_file is not None:
            shape_list = [args.shape_file]

        # Enable mixed precision if GPU is being used and AMP is available
        use_amp = hasattr(args, 'use_amp') and args.use_amp and args.gpu >= 0
        
        self.predict(model=model, data_loader=test_loader, 
                    output_bpseq=args.bpseq, output_bpp=args.bpp,
                    result=args.result, use_constraint=args.use_constraint,
                    shape_list=shape_list,
                    shape_intercept=args.shape_intercept, shape_slope=args.shape_slope,
                    use_amp=use_amp, task=args.task)


    def load_shape_reactivity(self, fname: str, intercept: float = -0.8, slope: float = 2.6) -> torch.tensor:
        r = []
        with open(fname) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l) == 2:
                    idx, val = l
                elif len(l) == 3:
                    idx, _, val = l
                else:
                    raise(ValueError(f"Invalid SHAPE reactivity file: {fname}"))
                idx, val = int(idx), float(val)
                while len(r) < idx:
                    r.append(-999)
                r[idx-1] = val
        # Deigan’s pseudoenergy approach
        r = torch.tensor(r, dtype=float)
        not_na = r > -1
        r[torch.logical_not(not_na)] = 0
        r[not_na] = slope * torch.log(r[not_na]+1) + intercept
        return r

    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('predict', help='predict')
        # input
        subparser.add_argument('input', type=str,
                            help='FASTA-formatted file or list of BPseq files')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--threads', type=int, default=1, metavar='N',
                            help='the number of threads (default: 1)')
        subparser.add_argument('--param', type=str, default='',
                            help='file name of trained parameters') 
        subparser.add_argument('--use-constraint', default=False, action='store_true')
        subparser.add_argument('--result', type=str, default=None,
                            help='output the prediction accuracy if reference structures are given')
        subparser.add_argument('--bpseq', type=str, default=None,
                            help='output the prediction with BPSEQ format to the specified directory')
        subparser.add_argument('--bpp', type=str, default=None,
                            help='output the base-pairing probability matrix to the specified directory')
        subparser.add_argument('--shape', type=str, default=None, help='specify the file name that includes the list of SHAPE reactivity files')
        subparser.add_argument('--shape-file', type=str, default=None, help='specify the file name that includes SHAPE reactivity')
        subparser.add_argument('--shape-intercept', type=float, default=-0.8,
                            help='Specify an intercept used with SHAPE restraints. Default is -0.8 kcal/mol.')
        subparser.add_argument('--shape-slope', type=float, default=2.6, 
                            help='Specify a slope used with SHAPE restraints. Default is 2.6.')
        subparser.add_argument('--use-amp', action='store_true',
                            help='use automatic mixed precision (AMP) for faster inference on GPUs')

        cls.add_task_args(subparser)

        cls.add_fold_args(subparser)
        cls.add_network_args(subparser)

        subparser.set_defaults(func = lambda args, conf: Predict().run(args, conf))
