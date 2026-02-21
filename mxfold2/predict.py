from __future__ import annotations

import logging
import os
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional
import pandas as pd

import torch
import torch.nn as nn

# import torch.nn.functional as F
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from mxfold2 import interface  # ty:ignore[unresolved-import]
from mxfold2.compbpseq import accuracy, compare_bpseq
from mxfold2.dataset import BPseqDataset, FastaDataset, RibonanzaDataset
from mxfold2.fold.fold import AbstractFold
from mxfold2.common import Common


def find_common_base(paths: list[str]) -> str | None:
    if not paths:
        return None
    abs_paths = [os.path.abspath(p) for p in paths]
    common = os.path.commonpath(abs_paths)
    if os.path.isfile(common):
        common = os.path.dirname(common)
    return common


def get_output_bpseq_path(
    header: str, base_dir: str | None, output_dir: str
) -> tuple[str, str]:
    if base_dir is not None and os.path.exists(header):
        abs_header = os.path.abspath(header)
        rel_path = os.path.relpath(abs_header, base_dir)
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        out_rel_path = rel_path_no_ext + ".bpseq"
        out_path = os.path.join(output_dir, out_rel_path)
    else:
        fn = os.path.basename(header)
        fn = os.path.splitext(fn)[0]
        out_rel_path = fn + ".bpseq"
        out_path = os.path.join(output_dir, out_rel_path)
    return out_path, out_rel_path


class Predict(Common):
    def __init__(self):
        super(Predict, self).__init__()

    def predict(
        self,
        model: AbstractFold | AveragedModel,
        data_loader: DataLoader,
        shape_model: Optional[list[nn.Module]] = None,
        output_bpseq: Optional[str] = None,
        bpseq_file: Optional[str] = None,
        output_bpp: Optional[str] = None,
        result: Optional[str] = None,
        use_constraint: bool = False,
        shape_list: Optional[list[Optional[str]]] = None,
        shape_intercept: float = 0.0,
        shape_slope: float = 0.0,
        use_amp: bool = False,
    ) -> None:

        res_fn = open(result, "w") if result is not None else None
        shape_list = [None] * len(data_loader) if shape_list is None else shape_list
        while len(shape_list) < len(data_loader):
            shape_list.append(None)

        model.eval()
        if shape_model is not None:
            for sm in shape_model:
                sm.eval()

        base_dir: str | None = None
        output_lst_f = None
        if output_bpseq is not None and output_bpseq != "stdout":
            headers_all = [
                data_loader.dataset[i][0] for i in range(len(data_loader.dataset))
            ]
            base_dir = find_common_base(headers_all)
            os.makedirs(output_bpseq, exist_ok=True)
            output_lst_path = os.path.join(output_bpseq, "output.lst")
            output_lst_f = open(output_lst_path, "w")

        seq_processed = 0
        with torch.no_grad():
            for headers, seqs, vals in data_loader:
                start = time.time()
                if use_constraint:
                    constraint = [
                        tgt if tp == "BPSEQ" else None
                        for tp, tgt in zip(vals["type"], vals["target"])
                    ]
                else:
                    constraint = None
                pseudoenergy = [
                    self.load_shape_reactivity(shape_file, shape_intercept, shape_slope)
                    if shape_file is not None
                    else None
                    for shape_file in shape_list[
                        seq_processed : seq_processed + len(seqs)
                    ]
                ]
                seq_processed += len(seqs)

                # Use autocast for mixed precision inference
                with autocast(
                    device_type=self.device_type, dtype=torch.float16, enabled=use_amp
                ):
                    if output_bpp is None:
                        scs, preds, bps = model(
                            seqs, constraint=constraint, pseudoenergy=pseudoenergy
                        )
                        pfs = bpps = [None] * len(preds)
                    else:
                        scs, preds, bps, pfs, bpps = model(
                            seqs,
                            return_partfunc=True,
                            constraint=constraint,
                            pseudoenergy=pseudoenergy,
                        )
                elapsed_time = time.time() - start
                for batch_j, (header, seq, ref, sc, pred, bp, pf, bpp) in enumerate(
                    zip(headers, seqs, vals["target"], scs, preds, bps, pfs, bpps)
                ):
                    if shape_model is not None:
                        p = []
                        for i, j in enumerate(bp):
                            if j == 0:
                                p.append([0, 0, 1])
                            elif i < j:
                                p.append([1, 0, 0])
                            elif i > j:
                                p.append([0, 1, 0])
                            else:
                                raise RuntimeError("unreachable")
                        p = torch.tensor(p, dtype=torch.float32, device=sc.device)
                        if "dataset_id" in vals:
                            shapes = [
                                shape_model[vals["dataset_id"][batch_j]].predict(seq, p)
                            ]
                        else:
                            shapes = [sm.predict(seq, p) for sm in shape_model]
                    if output_bpseq is None:
                        print(">" + header)
                        print(seq)
                        print(pred, f"({sc:.1f})")
                        if shape_model is not None:
                            for i, s in enumerate(zip(*shapes)):
                                print(f"{i + 1},", ", ".join(str(ss) for ss in s))
                    elif output_bpseq == "stdout":
                        print(f"# {header} (s={sc:.1f}, {elapsed_time:.5f}s)")
                        for i in range(1, len(bp)):
                            print(f"{i}\t{seq[i - 1]}\t{bp[i]}")
                    else:
                        out_path, out_rel_path = get_output_bpseq_path(
                            header, base_dir, output_bpseq
                        )
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with open(out_path, "w") as f:
                            print(
                                f"# {header} (s={sc:.1f}, {elapsed_time:.5f}s)", file=f
                            )
                            for i in range(1, len(bp)):
                                print(f"{i}\t{seq[i - 1]}\t{bp[i]}", file=f)
                        if output_lst_f is not None:
                            output_lst_f.write(
                                os.path.join(output_bpseq, out_rel_path) + "\n"
                            )
                    if bpseq_file is not None:
                        seq_index = seq_processed - len(seqs) + batch_j
                        if seq_index > 0:
                            logging.warning(
                                f"Multiple sequences detected, appending to {bpseq_file}"
                            )
                        mode = "a" if seq_index > 0 else "w"
                        with open(bpseq_file, mode) as f:
                            print(
                                f"# {header} (s={sc:.1f}, {elapsed_time:.5f}s)", file=f
                            )
                            for i in range(1, len(bp)):
                                print(f"{i}\t{seq[i - 1]}\t{bp[i]}", file=f)
                    if res_fn is not None:
                        if shape_model is not None and len(ref) > 0:
                            s = torch.tensor(shapes[0])
                            s = s.clip(min=0.0, max=1.0)
                            ref = ref[1:]
                            valid = ref >= -1
                            ref = ref.clip(min=0.0, max=1.0)
                            d = torch.abs(ref[valid] - s[valid])
                            d2 = d**2
                            x = [
                                header,
                                len(seq),
                                elapsed_time,
                                sc.item(),
                                len(d),
                                float(torch.sum(d) / len(d)),
                                float(torch.sum(d2) / len(d)),
                            ]
                            res_fn.write(", ".join([str(v) for v in x]) + "\n")
                        else:
                            x = compare_bpseq(ref, bp)
                            x = (
                                [header, len(seq), elapsed_time, sc.item()]
                                + list(x)
                                + list(accuracy(*x))
                            )
                            res_fn.write(", ".join([str(v) for v in x]) + "\n")
                    if output_bpp is not None:
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0]
                        fn = os.path.join(output_bpp, fn + ".bpp")
                        with open(fn, "w") as f:
                            for i in range(1, len(bpp)):
                                print(f"{i} {seq[i - 1]} ", end="", file=f)
                                for j, p in bpp[i]:
                                    print(f"{j}:{p:.3f}", end=" ", file=f)
                                print(file=f)

        if output_lst_f is not None:
            output_lst_f.close()

    def run(self, args: Namespace, conf: Optional[str] = None) -> None:
        torch.set_num_threads(args.threads)
        interface.set_num_threads(args.threads)

        convert_flag = getattr(args, "convert_t_to_u", False)
        if args.ribonanza:
            test_dataset = RibonanzaDataset(
                args.input, convert_t_to_u_flag=convert_flag
            )
        else:
            test_dataset = FastaDataset(args.input, convert_t_to_u_flag=convert_flag)
            if len(test_dataset) == 0:
                test_dataset = BPseqDataset(
                    args.input, convert_t_to_u_flag=convert_flag
                )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        model, _ = self.build_model(args)
        if args.param != "":
            param = Path(args.param)
            if not param.exists() and conf is not None:
                param = Path(conf).parent / param
            p = torch.load(param, map_location="cpu")
            if isinstance(p, dict) and "ema_state_dict" in p:
                p = p["ema_state_dict"]
            if isinstance(p, dict) and "model_state_dict" in p:
                p = p["model_state_dict"]
            if "n_averaged" in p:
                model = AveragedModel(model)
            model.load_state_dict(p)

        shape_model = None
        if args.shape_param and args.shape_param != "":
            param = Path(args.shape_param)
            if not param.exists() and conf is not None:
                param = Path(conf).parent / param
            p = torch.load(param, map_location="cpu")
            if isinstance(p, dict) and "shape_model_state_dict" in p:
                p = p["shape_model_state_dict"]
            shape_model = [self.build_shape_model(args) for _ in range(len(p))]
            for i, sm in enumerate(shape_model):
                sm.load_state_dict(p[i])

        device, self.device_type = self.get_device(args.gpu)
        model.to(device)
        if shape_model is not None:
            for sm in shape_model:
                sm.to(device)

        shape_list = None
        if args.shape is not None:
            shape_list = []
            with open(args.shape) as f:
                for l in f:
                    l = l.rstrip("\n").split()
                    shape_list.append(l[0])
        elif args.shape_file is not None:
            shape_list = [args.shape_file]

        # Enable mixed precision if GPU is being used and AMP is available
        use_amp = (
            hasattr(args, "use_amp")
            and args.use_amp
            and self.device_type in ("cuda", "mps")
        )

        self.predict(
            model=model,
            shape_model=shape_model,
            data_loader=test_loader,
            output_bpseq=args.bpseq,
            bpseq_file=getattr(args, "bpseq_file", None),
            output_bpp=args.bpp,
            result=args.result,
            use_constraint=args.use_constraint,
            shape_list=shape_list,
            shape_intercept=args.shape_intercept,
            shape_slope=args.shape_slope,
            use_amp=use_amp,
        )

    def load_shape_reactivity(
        self, fname: str, intercept: float = -0.8, slope: float = 2.6
    ) -> torch.Tensor:
        r = []
        with open(fname) as f:
            for l in f:
                l = l.rstrip("\n").split()
                if len(l) == 2:
                    idx, val = l
                elif len(l) == 3:
                    idx, _, val = l
                else:
                    raise (ValueError(f"Invalid SHAPE reactivity file: {fname}"))
                idx, val = int(idx), float(val)
                while len(r) < idx:
                    r.append(-999)
                r[idx - 1] = val
        # Deigan's pseudoenergy approach
        r = torch.tensor(r, dtype=torch.float)
        not_na = r > -1
        r[torch.logical_not(not_na)] = 0
        r[not_na] = slope * torch.log(r[not_na] + 1) + intercept
        return r

    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser("predict", help="predict")
        # input
        subparser.add_argument(
            "input", type=str, help="FASTA-formatted file or list of BPseq files"
        )
        subparser.add_argument(
            "--seed", type=int, default=0, metavar="S", help="random seed (default: 0)"
        )
        subparser.add_argument(
            "--gpu",
            type=int,
            default=-1,
            help="use GPU with the specified ID (default: -1 = CPU)",
        )
        subparser.add_argument(
            "--threads",
            type=int,
            default=1,
            metavar="N",
            help="the number of threads (default: 1)",
        )
        subparser.add_argument(
            "--param", type=str, default="", help="file name of trained parameters"
        )
        subparser.add_argument("--use-constraint", default=False, action="store_true")
        subparser.add_argument(
            "--result",
            type=str,
            default=None,
            help="output the prediction accuracy if reference structures are given",
        )
        subparser.add_argument(
            "--bpseq",
            type=str,
            default=None,
            help="output the prediction with BPSEQ format to the specified directory",
        )
        subparser.add_argument(
            "--bpseq-file",
            type=str,
            default=None,
            help="output the prediction to a single BPSEQ file (appends all sequences)",
        )
        subparser.add_argument(
            "--bpp",
            type=str,
            default=None,
            help="output the base-pairing probability matrix to the specified directory",
        )
        subparser.add_argument(
            "--shape",
            type=str,
            default=None,
            help="specify the file name that includes the list of SHAPE reactivity files",
        )
        subparser.add_argument(
            "--shape-file",
            type=str,
            default=None,
            help="specify the file name that includes SHAPE reactivity",
        )
        subparser.add_argument(
            "--shape-intercept",
            type=float,
            default=-0.8,
            help="Specify an intercept used with SHAPE restraints. Default is -0.8 kcal/mol.",
        )
        subparser.add_argument(
            "--shape-slope",
            type=float,
            default=2.6,
            help="Specify a slope used with SHAPE restraints. Default is 2.6.",
        )
        subparser.add_argument(
            "--shape-param",
            type=str,
            help="predict SHAPE reactivity using the SHAPE model with this parameter set.",
        )
        subparser.add_argument("--ribonanza", action="store_true")
        subparser.add_argument(
            "--convert-t-to-u",
            action="store_true",
            help="convert T to U in input sequences",
        )
        subparser.add_argument(
            "--use-amp",
            action="store_true",
            help="use automatic mixed precision (AMP) for faster inference on GPUs",
        )

        cls.add_fold_args(subparser)
        cls.add_network_args(subparser)

        subparser.set_defaults(func=lambda args, conf: Predict().run(args, conf))
