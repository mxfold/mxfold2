from __future__ import annotations

import logging
import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fold.fold import AbstractFold
from ..fold.shape_layers import _mask_and_clip_targets
from .implicit_mle import AdaptiveStepConfig, estimate_implicit_mle


class ShapeRankLoss(nn.Module):
    """
    Non-parametric, rank-based SHAPE loss using the implicit-MLE framework.

    - 1回目の fold で予測構造 pred_bps を得る
    - pred_bps から paired ベクトル (0/1, requires_grad=True) を作る
    - SHAPE targets の大小関係を教師として，低SHAPE位置のpairednessが
      高SHAPE位置より大きくなるようペアワイズlogistic lossを計算する
    - rank loss の dL/dpaired を pseudoenergy に変換する
    - forward または central finite difference で count 差分を計算する
    - 必要なら摂動幅 nu を paired-state flip率に合わせて配列ごとに調整する
    - 複数の coupled perturb-and-MAP 標本を平均して構造モデルへ勾配を流す
    """

    def __init__(
        self,
        model: AbstractFold,
        perturb: float = 0.0,
        nu: float = 0.1,
        tau: float = 0.2,
        l1_weight: float = 0.0,
        l2_weight: float = 0.0,
        sl_weight: float = 0.0,
        margin: float = 0.0,
        rank_delta: float = 0.0,
        alignment: str = "aligned",
        normalization: str = "none",
        estimator_scale: str = "divide_nu",
        clip_targets: bool = False,
        difference_mode: str = "forward",
        num_samples: int = 1,
        adaptive: AdaptiveStepConfig | None = None,
        pseudo_fy_weight: float = 0.0,
        weight_schedule: str = "none",
        weight_schedule_start: int = 1,
        weight_schedule_end: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.perturb = perturb
        self.nu = nu          # pseudoenergy のスケール
        self.tau = tau        # 順位ロスの温度
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.sl_weight = sl_weight
        self.margin = margin
        self.rank_delta = rank_delta
        self.alignment = alignment
        self.normalization = normalization
        self.estimator_scale = estimator_scale
        self.clip_targets = clip_targets
        self.difference_mode = difference_mode
        self.num_samples = num_samples
        self.adaptive = adaptive or AdaptiveStepConfig()
        self.pseudo_fy_weight = pseudo_fy_weight
        self.weight_schedule = weight_schedule
        self.weight_schedule_start = weight_schedule_start
        self.weight_schedule_end = weight_schedule_end
        self.current_epoch = 1
        self.total_epochs = 1
        self.last_imle_stats: dict[str, list[float]] = {}
        if tau <= 0.0:
            raise ValueError("rank tau must be positive")
        if rank_delta < 0.0:
            raise ValueError("rank delta must be non-negative")
        if sl_weight > 0.0 or pseudo_fy_weight > 0.0:
            from .. import param_turner2004
            from ..fold.rnafold import RNAFold

            self.turner = RNAFold(param_turner2004).to(
                next(self.model.parameters()).device
            )

    def set_epoch_info(self, epoch: int, total_epochs: int) -> None:
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def _get_reference_model_weights(self) -> dict[str, float] | None:
        if self.weight_schedule == "none" or not hasattr(self.model, "score_weight_turner"):
            return None

        schedule_end = self.weight_schedule_end or self.total_epochs
        if schedule_end <= self.weight_schedule_start:
            progress = float(self.current_epoch >= schedule_end)
        elif self.current_epoch < self.weight_schedule_start:
            progress = 0.0
        elif self.current_epoch >= schedule_end:
            progress = 1.0
        else:
            progress = (self.current_epoch - self.weight_schedule_start) / (
                schedule_end - self.weight_schedule_start
            )
        if self.weight_schedule == "cosine":
            progress = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.weight_schedule != "linear":
            raise ValueError(f"unknown weight schedule: {self.weight_schedule}")

        return {
            "score_weight_turner": 1.0 - 0.5 * progress,
            "score_weight_positional": 0.5 * progress,
        }

    def _per_sequence_rank_loss(
        self,
        shape_vec: torch.Tensor,
        paired_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        1本の配列についての順位ロスを計算する。

        shape_vec: [L_all]  (raw SHAPE targets, 1D)
        paired_vec: [L_all]  (0/1, same length as pred_bps / sequence)
        """
        device = paired_vec.device

        if self.clip_targets:
            valid, t_all = _mask_and_clip_targets(
                shape_vec, min_val=0.01, max_val=1.0
            )
        else:
            valid = shape_vec > -10
            t_all = shape_vec

        # valid な点が全く無い場合：loss=0だが paired 依存にして勾配経路だけ残す
        if not valid.any():
            return paired_vec.sum() * 0.0

        t_valid = t_all[valid].to(device)
        p_mask = paired_vec[valid].to(device)

        # 要素数が 1 以下ではSHAPEの大小関係を作れない。
        if t_valid.numel() <= 1:
            return p_mask.sum() * 0.0

        # shape_gap[i,j] > 0 なら i の方が低SHAPEなので、
        # pairedness[i] - pairedness[j] が正になることを教師信号にする。
        # 比較集合は予測構造に依存しないため、全unpaired/全pairedでもlossと
        # 勾配が定義される。SHAPE差がrank_delta以下の曖昧な組は除外する。
        shape_gap = t_valid[None, :] - t_valid[:, None]
        ordered = shape_gap > self.rank_delta
        if not ordered.any():
            return p_mask.sum() * 0.0

        paired_gap = p_mask[:, None] - p_mask[None, :]
        pairwise_loss = F.softplus((self.margin - paired_gap) / self.tau)
        return pairwise_loss[ordered].mean()

    def forward(
        self,
        seq: List[str],
        targets: List[torch.Tensor],
        fname: Optional[List[str]] = None,
        dataset_id: Optional[int] = None,
    ) -> torch.Tensor:
        # dataset_id は ShapeNLLLoss との互換性のために受け取るが，ここでは使わない
        del dataset_id

        def objective(paired: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(
                [
                    self._per_sequence_rank_loss(target.to(paired[0].device), pair)
                    for target, pair in zip(targets, paired)
                ]
            )

        estimate = estimate_implicit_mle(
            self.model,
            seq,
            objective,
            perturb=self.perturb,
            nu=self.nu,
            alignment=self.alignment,
            normalization=self.normalization,
            estimator_scale=self.estimator_scale,
            difference_mode=self.difference_mode,
            num_samples=self.num_samples,
            adaptive=self.adaptive,
            reference_model_weights=self._get_reference_model_weights(),
        )
        loss = estimate.loss
        self.last_imle_stats = {
            "selected_nu": estimate.selected_nu,
            "flip_fraction": estimate.flip_fraction,
        }

        l = torch.tensor([len(s) for s in seq], device=estimate.loss.device)
        if self.pseudo_fy_weight > 0.0:
            with torch.no_grad():
                _, _, turner_structures = self.turner(seq)
            fy_ref, _, _ = self.model(
                seq,
                constraint=turner_structures,
                max_internal_length=None,
            )
            loss = loss + self.pseudo_fy_weight * (
                estimate.prediction_score - fy_ref
            ) / l

        if self.sl_weight > 0.0:
            with torch.no_grad():
                ref2, _, _ = self.turner(
                    seq,
                    constraint=estimate.target_structures,
                )
            loss = loss + self.sl_weight * (estimate.target_score - ref2) ** 2 / l

        # logging は scalar 前提なので，バッチサイズ >1 のときは mean を取っておく
        scalar_loss = loss.mean()
        logging.debug(
            "ShapeRankLoss = %s = (%s - %s)",
            scalar_loss.item(),
            estimate.prediction_score.mean().item(),
            estimate.target_score.mean().item(),
        )
        logging.debug(seq)
        logging.debug(estimate.prediction_structures)
        logging.debug(estimate.target_structures)
        if float(scalar_loss.item()) > 1e10 or torch.isnan(scalar_loss):
            logging.error(fname)
            logging.error(
                "%s, %s, %s",
                scalar_loss.item(),
                estimate.prediction_score.mean().item(),
                estimate.target_score.mean().item(),
            )
            logging.error(seq)

        # l1/l2 正則化が欲しければここで追加（デフォルトは 0）
        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                scalar_loss = scalar_loss + self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg = l2_reg + torch.sum(p ** 2)
        #     scalar_loss = scalar_loss + self.l2_weight * l2_reg

        return scalar_loss
