from __future__ import annotations

import logging
import math
from typing import Optional

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

def _mask_and_clip_targets(t: torch.Tensor, min_val: float = 0.01, max_val: float = 1.0):
    """
    共通処理:
      - t の要素ごとに t < -1 は invalid (False) とする（スルー）
      - それ以外は [min_val, max_val] に clip する
    戻り値: (valid_mask (bool tensor), clipped_tensor)
    """
    # assume t is a torch.Tensor (as used in callers). make a safe clone to avoid in-place issues
    if not torch.is_tensor(t):
        t = torch.tensor(t)
    t = t.clone()
    valid = t > -10
    # clip entire tensor (clipped values will be ignored where valid is False)
    clipped = torch.clamp(t, min=min_val, max=max_val)
    return valid, clipped


class GeneralizedExtremeValue(torch.autograd.Function):
    def __init__(self, 
                xi: float | torch.tensor,
                mu: float | torch.tensor, 
                sigma: float | torch.tensor):
        self.xi = xi
        self.mu = mu
        self.sigma = sigma
    
    def log_prob(self, x: torch.tensor) -> torch.tensor:
        z = 1 + self.xi / self.sigma * (x - self.mu)
        # Evaluate the clipped log density directly. Outside the GEV support,
        # or close enough to its boundary that the density is below 1e-5, the
        # native Wu implementation clips to that same density floor.
        log_y = -self.xi.reciprocal() * torch.log(
            z.clamp_min(torch.finfo(z.dtype).tiny)
        )
        safe_log_y = log_y.clamp(max=10.0)
        log_density = (
            -torch.log(self.sigma)
            + (1 + self.xi) * safe_log_y
            - torch.exp(safe_log_y)
        )
        floor = math.log(1e-5)
        in_support = (z > 0) & (log_y <= 10.0)
        return torch.where(
            in_support,
            log_density.clamp_min(floor),
            torch.full_like(log_density, floor),
        )


class Wu(nn.Module):
    def __init__(self,  
            xi: float = 0.774, 
            mu: float = 0.078, 
            sigma: float = 0.083,
            alpha: float = 1.006, 
            beta: float = 1.404,
            ) -> None:
        super(Wu, self).__init__()
        self.xi = nn.Parameter(torch.tensor(xi))
        self.mu = nn.Parameter(torch.tensor(mu))
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.paired_dist = GeneralizedExtremeValue(self.xi, self.mu, self.sigma)
        self.unpaired_dist = Gamma(self.alpha, self.beta)


    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
        self.xi.data.clamp_(min=1e-2, max=2.0)
        self.sigma.data.clamp_(min=1e-2, max=2.0)
        self.mu.data.clamp_(min=1e-2, max=2.0)
        self.alpha.data.clamp_(min=1e-2, max=5.0)
        self.beta.data.clamp_(min=1e-2, max=5.0)

        device = self.xi.device

        nlls = []
        for i in range(len(seq)):
            valid, t_all = _mask_and_clip_targets(targets[i], min_val=0.01, max_val=1.0)
            if not valid.any():
                nlls.append(torch.tensor(0.0, device=device))
                continue
            t = t_all[valid].to(device)
            p = paired[i][valid].to(t.dtype).to(device)

            nll = -torch.mean(self.paired_dist.log_prob(t) * p
                              + self.unpaired_dist.log_prob(t) * (1 - p))
            nlls.append(nll)
        
        # --- 配列単位で NaN チェック ---
        if torch.isnan(torch.stack(nlls)).any():
            logging.error("[NaN detected in batch]")
            logging.error(f"xi={self.xi.item():.4f}, mu={self.mu.item():.4f}, "
                            f"sigma={self.sigma.item():.4f}, alpha={self.alpha.item():.4f}, beta={self.beta.item():.4f}")
            # logging.error(f"nlls={nlls.detach().cpu().numpy()}")

        return torch.stack(nlls)

class Foo(nn.Module):
    def __init__(self,  
            p_alpha: float = 0.540,
            p_beta: float = 1.390,
            u_alpha: float = 1.006, 
            u_beta: float = 1.404,
            ) -> None:
        super(Foo, self).__init__()
        self.p_alpha = nn.Parameter(torch.tensor(p_alpha))
        self.p_beta = nn.Parameter(torch.tensor(p_beta))
        self.u_alpha = nn.Parameter(torch.tensor(u_alpha))
        self.u_beta = nn.Parameter(torch.tensor(u_beta))
        self.paired_dist = Gamma(self.p_alpha, self.p_beta)
        self.unpaired_dist = Gamma(self.u_alpha, self.u_beta)

    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
         self.p_alpha.data.clamp_(min=1e-2)
         self.p_beta.data.clamp_(min=1e-2)
         self.u_alpha.data.clamp_(min=1e-2)
         self.u_beta.data.clamp_(min=1e-2)

         nlls = []
         for i in range(len(seq)):
            valid, t_all = _mask_and_clip_targets(targets[i], min_val=0.01, max_val=1.0)
            if not valid.any():
                nlls.append(torch.tensor(0.0, device=self.p_alpha.device))
                continue
            t = t_all[valid].to(self.p_alpha.device)
            p = paired[i][valid].to(t.dtype).to(self.p_alpha.device)
            nll = -torch.mean(self.paired_dist.log_prob(t) * p
                              + self.unpaired_dist.log_prob(t) * (1 - p))
            nlls.append(nll)
         return torch.stack(nlls)


# class RiboEM(nn.Module):
#     """
#     log1p 空間での 2 成分ガウス（paired / unpaired）から尤度を計算するクラス。
#     Wu と同じ forward(seq, paired, targets) シグネチャを持ち、初期値をここに直接指定します。
#     """
#     def __init__(self,
#                  mu_u: float = 0.36248604585583205,
#                  sig_u: float = 0.3004844655699528,
#                  mu_p: float = 0.0,
#                  sig_p: float = 0.10) -> None:
#         super(RiboEM, self).__init__()
#         # Wu と同じくパラメータを nn.Parameter として保持（必要に応じて学習可能に）
#         self.mu_u = nn.Parameter(torch.tensor(mu_u))
#         self.sig_u = nn.Parameter(torch.tensor(sig_u))
#         self.mu_p = nn.Parameter(torch.tensor(mu_p))
#         self.sig_p = nn.Parameter(torch.tensor(sig_p))

#     def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
#         # 安定化のため clamp（Wu と同様の扱い）
#         self.sig_u.data.clamp_(min=1e-6, max=10.0)
#         self.sig_p.data.clamp_(min=1e-6, max=10.0)
#         self.mu_u.data.clamp_(min=-10.0, max=10.0)
#         self.mu_p.data.clamp_(min=-10.0, max=10.0)
 
#         nlls = []
#         two_pi = torch.tensor(2.0 * np.pi)
 
#         for i in range(len(seq)):
#             # Wu と同じ基準で無効値を除外 -> 統一処理
#             valid, t_all = _mask_and_clip_targets(targets[i], min_val=0.01, max_val=1.0)
#             if not valid.any():
#                 nlls.append(torch.tensor(0.0, device=self.mu_u.device))
#                 continue
#             t = t_all[valid].to(self.mu_u.device)
#             p = paired[i][valid].to(t.dtype).to(self.mu_u.device)

#             device = t.device
#             mu_u = self.mu_u.to(device)
#             sig_u = (self.sig_u.to(device) + 1e-12)
#             mu_p = self.mu_p.to(device)
#             sig_p = (self.sig_p.to(device) + 1e-12)
 
#             z = torch.log1p(t)

#             log_const_u = -0.5 * torch.log(two_pi.to(device)) - torch.log(sig_u)
#             log_pdf_u = log_const_u - 0.5 * ((z - mu_u) / sig_u) ** 2

#             log_const_p = -0.5 * torch.log(two_pi.to(device)) - torch.log(sig_p)
#             log_pdf_p = log_const_p - 0.5 * ((z - mu_p) / sig_p) ** 2

#             nll = -torch.mean(log_pdf_u * (1 - p) + log_pdf_p * p)
#             nlls.append(nll)

#         if torch.isnan(torch.stack(nlls)).any():
#             logging.error("[NaN detected in RiboEM batch]")
#             logging.error(f"mu_u={self.mu_u.item():.4f}, sig_u={self.sig_u.item():.4f}, "
#                           f"mu_p={self.mu_p.item():.4f}, sig_p={self.sig_p.item():.4f}")

#         return torch.stack(nlls)

class ContraSE(nn.Module):
    """
    Base-specific Gamma parameters hardcoded from EternaFoldParams_PLUS_POTENTIALS.v1.

    Mapping:
      - k -> k (Eterna の k, ここではパラメータ名に k を使う)
      - theta -> theta (Eterna の theta, ここではパラメータ名に theta を使う)
      - suffix 0 = paired, 1 = unpaired
    """
    def __init__(self) -> None:
        super().__init__()
        from collections import OrderedDict
        import math

        # Values taken from EternaFoldParams_PLUS_POTENTIALS.v1 (embedded)
        eterna_vals = {
            "A": {"k0": 0.3146640447, "k1": 0.997568854,  "th0": -2.118602351, "th1": -0.9497727393},
            "C": {"k0": 0.4536549189, "k1": 0.7960273654, "th0": -2.525121753, "th1": -1.598290642},
            "G": {"k0": 0.3607764847, "k1": 0.7175884215, "th0": -2.065437383, "th1": -0.9349060146},
            "U": {"k0": 0.3820697829, "k1": 0.6217085081, "th0": -2.391501329, "th1": -0.8960615316},
        }

        self.bases = ["A", "U", "G", "C"]
        params = OrderedDict()
        for b in self.bases:
            ev = eterna_vals[b]
            # suffix 0 -> paired, suffix 1 -> unpaired
            p_k = float(ev["k0"])
            p_theta = float(ev["th0"])
            u_k = float(ev["k1"])
            u_theta = float(ev["th1"])

            # store as nn.Parameter with names using k/theta
            params[f"p_{b}_k"] = nn.Parameter(torch.tensor(p_k, dtype=torch.float32))
            params[f"p_{b}_theta"]  = nn.Parameter(torch.tensor(p_theta, dtype=torch.float32))
            params[f"u_{b}_k"] = nn.Parameter(torch.tensor(u_k, dtype=torch.float32))
            params[f"u_{b}_theta"]  = nn.Parameter(torch.tensor(u_theta, dtype=torch.float32))

        self.params = nn.ParameterDict(params)

    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
        # clamp parameters for stability
        for name, p in self.params.items():
            # k should be positive-ish, theta can be negative; clamp magnitudes
            if name.endswith("_k"):
                p.data.clamp_(min=1e-6, max=50.0)
            else:
                p.data.clamp_(min=-10.0, max=10.0)
 
        device = next(self.params.values()).device
        nlls = []
 
        # unified handling: mask & clip targets per-sample, build bases_at_valid from valid indices,
        # then compute per-base contributions indexing into the valid-subset tensors.
        for i in range(len(seq)):
            valid, t_all = _mask_and_clip_targets(targets[i], min_val=0.01, max_val=1.0)
            if not valid.any():
                nlls.append(torch.tensor(0.0, device=device))
                continue

            # t_valid is clipped values at valid positions (1D)
            t_valid = t_all[valid].to(device)
            p_mask = paired[i][valid].to(t_valid.dtype).to(device)

            # map valid indices back to sequence positions to build bases_at_valid
            idxs = torch.where(valid)[0].cpu().numpy().tolist()
            seq_i = seq[i]
            if len(targets[i]) == len(seq_i) + 1:
                bases_at_valid = [seq_i[j-1] if j > 0 else "N" for j in idxs]
            else:
                bases_at_valid = [seq_i[j] for j in idxs]

            total_logp = torch.tensor(0.0, device=device)
            total_count = 0

            for b in self.bases:
                # positions of this base within the valid-subset
                idx_list = [k for k, ch in enumerate(bases_at_valid) if ch.upper() == b]
                if not idx_list:
                    continue
                idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
                t_b = t_valid[idx_tensor]
                p_b = p_mask[idx_tensor]

                # read k and theta, convert to Gamma params: alpha = exp(k), beta = exp(theta)
                k_p = self.params[f"p_{b}_k"].to(device)
                th_p = self.params[f"p_{b}_theta"].to(device)
                k_u = self.params[f"u_{b}_k"].to(device)
                th_u = self.params[f"u_{b}_theta"].to(device)
                #contrafold-seとpytorchでbetaの定義が違うので注意
                pa = torch.exp(k_p)
                pb = 1.0 / torch.exp(th_p)
                ua = torch.exp(k_u)
                ub = 1.0 / torch.exp(th_u)

                paired_dist = torch.distributions.Gamma(pa, pb)
                unpaired_dist = torch.distributions.Gamma(ua, ub)

                logp_b = paired_dist.log_prob(t_b) * p_b + unpaired_dist.log_prob(t_b) * (1.0 - p_b)
                total_logp = total_logp + torch.sum(logp_b)
                total_count += t_b.numel()

            if total_count == 0:
                nlls.append(torch.tensor(0.0, device=device))
            else:
                nll = - (total_logp / float(total_count))
                nlls.append(nll)

        # NaN チェック
        st = torch.stack(nlls)
        if torch.isnan(st).any():
            logging.error("[NaN detected in ContraSE batch]")
            # log k/theta values for debugging
            debug_vals = {k: float(v.item()) for k, v in self.params.items()}
            logging.error(debug_vals)
        return st


class Helix(nn.Module):
    """
    Helix context model.

    Context names (ユーザ指定):
      1: stacking  (i と i+1 が互いにペア)
      2: unpair    (i, i+1 ともにアンペア)
      3: opening   (i がアンペアで i+1 がペア)
      4: closing   (i がペアで i+1 がアンペア)

    Paired -> GEV(xi, mu, sigma)
    Unpaired -> Gamma(alpha, beta)

    optional wu_params を渡すと Wu のパラメータ値を直接使う（Wu モジュールは生成しない）。
    wu_params のキー: xi, mu, sigma, alpha, beta
    """
    def __init__(self, wu_params: dict | None = None) -> None:
        super().__init__()
        from collections import OrderedDict

        # contexts
        self.ctx_names = ["stacking", "unpair", "opening", "closing"]

        # デフォルト値（Wu の既知のデフォルトを用意）
        default_gev = {"xi": 0.774, "mu": 0.078, "sigma": 0.083}
        default_gamma = {"alpha": 1.006, "beta": 1.404}

        wp = wu_params or {}
        gev_vals = {
            "xi": float(wp.get("xi", default_gev["xi"])),
            "mu": float(wp.get("mu", default_gev["mu"])),
            "sigma": float(wp.get("sigma", default_gev["sigma"])),
        }
        gamma_vals = {
            "alpha": float(wp.get("alpha", default_gamma["alpha"])),
            "beta": float(wp.get("beta", default_gamma["beta"])),
        }

        params = OrderedDict()
        # 各コンテキストごとに paired は GEV パラメータ、unpaired は Gamma パラメータを持つ
        for ctx in self.ctx_names:
            params[f"{ctx}_p_xi"] = nn.Parameter(torch.tensor(gev_vals["xi"], dtype=torch.float32))
            params[f"{ctx}_p_mu"] = nn.Parameter(torch.tensor(gev_vals["mu"], dtype=torch.float32))
            params[f"{ctx}_p_sigma"] = nn.Parameter(torch.tensor(gev_vals["sigma"], dtype=torch.float32))
            params[f"{ctx}_u_alpha"] = nn.Parameter(torch.tensor(gamma_vals["alpha"], dtype=torch.float32))
            params[f"{ctx}_u_beta"] = nn.Parameter(torch.tensor(gamma_vals["beta"], dtype=torch.float32))

        self.params = nn.ParameterDict(params)

    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
        # clamp for stability
        for name, p in self.params.items():
            if name.endswith("_p_xi"):
                p.data.clamp_(min=1e-3, max=5.0)
            elif name.endswith("_p_sigma"):
                p.data.clamp_(min=1e-6, max=5.0)
            elif name.endswith("_p_mu"):
                p.data.clamp_(min=-10.0, max=10.0)
            elif name.endswith("_u_alpha"):
                p.data.clamp_(min=1e-6, max=50.0)
            elif name.endswith("_u_beta"):
                p.data.clamp_(min=1e-6, max=50.0)

        device = next(self.params.values()).device
        nlls = []

        for idx in range(len(seq)):
            seq_i = seq[idx]
            L = len(seq_i)
            targ_full = targets[idx]
            paired_full = paired[idx]
 
            offset = 1 if len(targ_full) == L + 1 else 0
 
            total_logp = torch.tensor(0.0, device=device)
            total_count = 0
 
            for i in range(0, L - 1):
                ti = i + offset
                ti1 = i + 1 + offset
                if ti >= len(targ_full) or ti1 >= len(targ_full):
                    continue
                # 統一処理: use mask-and-clip for this single position
                valid, t_all = _mask_and_clip_targets(targ_full[ti].unsqueeze(0), min_val=0.01, max_val=1.0)
                if not valid[0]:
                    continue
                t = t_all[0].to(device)
 
                p_i_raw = paired_full[ti]
                p_i1_raw = paired_full[ti1]

                if torch.is_floating_point(p_i_raw):
                    is_p_i = float(p_i_raw.item()) > 0.5
                    is_p_i1 = float(p_i1_raw.item()) > 0.5
                    stacking = is_p_i and is_p_i1 and False
                else:
                    p_i_idx = int(p_i_raw.item())
                    p_i1_idx = int(p_i1_raw.item())
                    is_p_i = p_i_idx != 0
                    is_p_i1 = p_i1_idx != 0
                    stacking = (p_i_idx == (i + 2)) and (p_i1_idx == (i + 1))

                if stacking:
                    ctx = "stacking"
                elif (not is_p_i) and (not is_p_i1):
                    ctx = "unpair"
                elif (not is_p_i) and is_p_i1:
                    ctx = "opening"
                else:
                    ctx = "closing"

                if is_p_i:
                    # paired -> GEV
                    xi = self.params[f"{ctx}_p_xi"].to(device)
                    mu = self.params[f"{ctx}_p_mu"].to(device)
                    sigma = self.params[f"{ctx}_p_sigma"].to(device)
                    gev = GeneralizedExtremeValue(xi, mu, sigma)
                    logp = gev.log_prob(t)
                else:
                    # unpaired -> Gamma
                    a = self.params[f"{ctx}_u_alpha"].to(device)
                    b = self.params[f"{ctx}_u_beta"].to(device)
                    dist = Gamma(a, b)
                    logp = dist.log_prob(t)

                total_logp = total_logp + logp
                total_count += 1

            if total_count == 0:
                nlls.append(torch.tensor(0.0, device=device))
            else:
                nll = - (total_logp / float(total_count))
                nlls.append(nll)

        st = torch.stack(nlls)
        if torch.isnan(st).any():
            logging.error("[NaN detected in Helix]")
            logging.error({k: float(v.item()) for k, v in self.params.items()})
        return st

class Corr(nn.Module):
    """
    Reactivity と二次構造（paired/unpaired）との相関に基づく損失。
    - targets に対して _mask_and_clip_targets() を適用して valid positions を選ぶ。
    - 二次構造は paired ->shape proxy 0.05, unpaired -> 1 として扱う（dot=1, bracket=0）。
    - 各シーケンスについて Pearson 相関 r を計算し、損失として 1 - r を返す。
      （valid が無ければ 0 を返す）
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, seq: list[str], paired: list[torch.tensor], targets: list[torch.Tensor]):
        import sys
        device = paired[0].device if len(paired) > 0 else (targets[0].device if len(targets)>0 else torch.device("cpu"))
        output = []
        
        for i in range(len(seq)):
            valid, t_all = _mask_and_clip_targets(targets[i], min_val=0.01, max_val=1.0)
            # 基本情報の出力
            logging.debug("Corr: idx=%d seq_len=%d target_len=%d valid_count=%d",
                          i, len(seq[i]), int(t_all.numel()), int(valid.sum().item() if hasattr(valid, "sum") else 0))

            if not valid.any():
                # 問題箇所としてログに詳細を出す（停止せず元の挙動を維持）
                logging.error("Corr: idx=%d NO VALID POSITIONS. seq='%s'", i, seq[i])
                try:
                    logging.error("  t_all (first 50)=%s", t_all.detach().cpu().tolist()[:50])
                except Exception:
                    logging.exception("  failed to dump t_all")
                try:
                    logging.error("  paired (first 50)=%s", paired[i].detach().cpu().tolist()[:50])
                except Exception:
                    logging.exception("  failed to dump paired[i]")
                print(f"Corr DEBUG idx={i} NO_VALID seq={seq[i]}", file=sys.stderr)
                output.append(torch.tensor(0.0, device=device))
                continue

            # 有効位置だけ取り出す
            t_valid = t_all[valid].to(device).to(torch.float32)
            p_full = paired[i]
            p_valid = p_full[valid].to(device).to(torch.float32)
            # map predicted paired-prob p_valid -> reactivity scale matching sim_shape.fake:
            # paired (p=1) -> 0.05, unpaired (p=0) -> 1.0
            shape_proxy = 1.0 - 0.95 * p_valid

            # ここで要素数不足が発生するか確認して詳細出力
            if shape_proxy.numel() < 2:
                logging.error("Corr: idx=%d TOO FEW VALID POSITIONS AFTER MASK: num=%d seq='%s'", i, int(shape_proxy.numel()), seq[i])
                try:
                    logging.error("  t_valid=%s", t_valid.detach().cpu().tolist())
                except Exception:
                    logging.exception("  failed to dump t_valid")
                try:
                    logging.error("  p_valid=%s", p_valid.detach().cpu().tolist())
                except Exception:
                    logging.exception("  failed to dump p_valid")
                print(f"Corr DEBUG idx={i} TOO_FEW num={shape_proxy.numel()} seq={seq[i]}", file=sys.stderr)
                output.append(torch.tensor(0.0, device=device))
                continue

            # Pearson 相関を計算
            x = shape_proxy
            y = t_valid
            x_mean = torch.mean(x)
            y_mean = torch.mean(y)
            xm = x - x_mean
            ym = y - y_mean
            cov = torch.mean(xm * ym)
            x_std = torch.sqrt(torch.mean(xm * xm))
            y_std = torch.sqrt(torch.mean(ym * ym))

            denom = x_std * y_std
            # 安定化 eps（dtype/device に合わせる）
            eps = torch.finfo(denom.dtype).eps * 1e3
            denom_safe = denom + eps
            corr_raw = cov / denom_safe
            # denom が小さいときは fallback（相関未定義） -> loss=1.0 にしたい
            small_mask = denom <= eps
            # 勾配経路を保つため p_valid を使ったゼロ（影響はゼロ）
            fallback_corr = p_valid.sum() * 0.0
            corr = torch.where(small_mask, fallback_corr, corr_raw)
            corr = torch.clamp(corr, -1.0, 1.0)
 
            loss = 1.0 - corr
            # denom が小さい箇所は明示的に最大損失(=1.0)。勾配経路は保つ。
            loss = torch.where(small_mask, torch.tensor(1.0, device=device, dtype=loss.dtype) + (p_valid.sum()*0.0), loss + (p_valid.sum()*0.0))
            output.append(loss)

        st = torch.stack(output)
        if torch.isnan(st).any():
            logging.error("[NaN detected in Corr]")
            try:
                logging.error(f"1-Corr={st.detach().cpu().numpy()}")
            except Exception:
                logging.exception("failed to dump 1-Corr")
        return st
