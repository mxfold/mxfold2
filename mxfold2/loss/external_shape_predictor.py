import torch
import torch.nn as nn
import sys
from pathlib import Path

# --- 外部SHAPEtransformerをパスに追加 ---
sys.path.append("/gs/bs/tga-satolab-gtex/yamauchi/SHAPEtransformer")
from train import ShapeTransformer
class ExternalShapePredictor(nn.Module):
    """
    外部 SHAPEtransformer モデルを読み込み、
    - predict(): shape reactivity 予測 (list[str], list[Tensor]) -> list[Tensor]
    - forward(): loss 計算 (list[str], list[Tensor], list[Tensor]) -> scalar loss
    """

    def __init__(self, model_path: str | Path = None, device: str | None = None, loss_type: str = "mse"):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.loss_type = loss_type

        # --- デフォルトモデルパス ---
        if model_path is None:
            model_path = Path("/gs/bs/tga-satolab-gtex/yamauchi/SHAPEtransformer/logs/cls-v70-CD80/model.pth")

        # --- チェックポイント読み込み（堅牢に） ---
        # map_location のみ指定して読み込む（weights_only は存在しない引数）
        ckpt = torch.load(model_path, map_location=self.device)

        # normalize checkpoint -> state and args
        if isinstance(ckpt, dict):
            ck_args = ckpt.get("args", {}) or {}
            # support common keys: "model_state_dict" or "model_state" or direct state dict
            if "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            elif "model_state" in ckpt:
                state = ckpt["model_state"]
            else:
                # ckpt may itself be the state dict
                state = {k: v for k, v in ckpt.items() if "weight" in k or "bias" in k} if not ckpt.get("args") else ckpt
                # fallback to ckpt if above heuristic fails
                if not state:
                    state = ckpt
        else:
            ck_args = {}
            state = ckpt

        # instantiate model using saved args (fallback defaults)
        # restore loss_type and cls params if present in checkpoint args
        loss_type = ck_args.get("loss_type", ck_args.get("loss-type", "mse"))
        try:
            cls_b0 = float(ck_args.get("cls_b0", ck_args.get("cls-b0", 0.25)))
            cls_b1 = float(ck_args.get("cls_b1", ck_args.get("cls-b1", 0.5)))
            cls_tau = float(ck_args.get("cls_tau", ck_args.get("cls-tau", 10.0)))
        except Exception:
            cls_b0, cls_b1, cls_tau = 0.25, 0.5, 10.0

        self.model = ShapeTransformer(
            d_model=int(ck_args.get("dim", 128)),
            nhead=int(ck_args.get("nhead", 8)),
            num_layers=int(ck_args.get("layers", 2)),
            dim_feedforward=int(ck_args.get("dim_feedforward", 256)),
            dropout=float(ck_args.get("dropout", 0.1)),
            max_len=int(ck_args.get("max_len", 2000)),
            loss_type=str(loss_type),
            cls_b0=cls_b0,
            cls_b1=cls_b1,
            cls_tau=cls_tau,
        ).to(self.device)

        loaded = False
        try:
            self.model.load_state_dict(state)
            loaded = True
        except Exception:
            if isinstance(state, dict):
                if "model_state_dict" in state:
                    self.model.load_state_dict(state["model_state_dict"])
                    loaded = True
                elif "model_state" in state:
                    self.model.load_state_dict(state["model_state"])
                    loaded = True
        if not loaded:
            raise RuntimeError(f"Failed to load model weights from {model_path}")

        # freeze and set eval (external predictor should act as fixed prior)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

    # helper: convert tensor of base indices -> sequence string (if needed)
    def _tensor_to_seq(self, t: torch.Tensor) -> str:
        t = t.detach().cpu().squeeze()
        idx_to_base = {0: "A", 1: "C", 2: "G", 3: "U", 4: "N", 5: "N"}
        return "".join(idx_to_base.get(int(x), "N") for x in t.tolist())

    # ------------------------------------------------------------
    # 前処理: 現在は seq は文字列期待、partner/target は index0 を含む1次元テンソル期待
    # ------------------------------------------------------------
    def _prepare_single(self, seq, paired, target=None):
        # seq: accept str or 1d-tensor-of-indices
        if isinstance(seq, str):
            seq_str = seq
        elif torch.is_tensor(seq):
            seq_str = self._tensor_to_seq(seq)
        else:
            raise TypeError("seq must be str or Tensor of base indices")

        # paired: expect 1D tensor length L+1 (index0 included) or convertible
        if torch.is_tensor(paired):
            # move to device but keep graph connectivity if caller enabled requires_grad
            paired_t = paired.to(self.device).squeeze()
        else:
            paired_t = torch.tensor(paired, device=self.device)

        # ensure 1D and proper dtype
        if paired_t.dim() == 0:
            paired_t = paired_t.unsqueeze(0)
        paired_t = paired_t.float()

        # target: if provided, keep index0 included
        if target is not None:
            if torch.is_tensor(target):
                target_t = target.to(self.device).squeeze()
            else:
                target_t = torch.tensor(target, device=self.device)
            if target_t.dim() == 0:
                target_t = target_t.unsqueeze(0)
        else:
            target_t = None

        return seq_str, paired_t, target_t

    # ------------------------------------------------------------
    # 推論モード: shape reactivity 予測
    # ------------------------------------------------------------
    @torch.no_grad()
    def predict(self, seq_list, paired_list):
        # seq_list: iterable of str or 1D-tensor
        # paired_list: iterable of 1D-tensor (index0 included)
        seqs = []
        paireds = []
        for seq, paired in zip(seq_list, paired_list):
            seq_s, paired_t, _ = self._prepare_single(seq, paired)
            seqs.append(seq_s)
            paireds.append(paired_t)

        # return predictions on the model/device (caller may .cpu() if desired)
        preds = self.model.predict(seqs, paireds)  # list[Tensor] on device
        return [p.detach() for p in preds]

    # ------------------------------------------------------------
    # 学習モード: loss 計算（MSE / MAE / CLS を選択）
    # ------------------------------------------------------------
    def forward(self, seq_list, paired_list, target_list, paired_requires_grad: bool = False):
        """
        常に paired に対して勾配を流す（fold 側で autograd を使うため）。
        外部モデルの predict() を使わず常に model.forward(...) を呼び出す。
        """
        seqs = []
        paireds = []
        targets = []
        for seq, paired, target in zip(seq_list, paired_list, target_list):
            seq_s, paired_t, target_t = self._prepare_single(seq, paired, target)

            # 常に勾配を得るため paired を require_grad にする
            paired_t = paired_t.clone().to(self.device)
            paired_t.requires_grad_(True)

            seqs.append(seq_s)
            paireds.append(paired_t)

            # ターゲットは index0 を含む想定。None は NaN ベクトルで埋める。
            if target_t is None:
                targets.append(None)
            else:
                targets.append(target_t.to(self.device).float().clamp(0.0, 1.0))

        # 常に model.forward を呼び出して scalar loss を得る（autograd 経路確保）
        # model.forward は (seqs, paireds, targets) -> scalar loss を返す想定
        proc_targets = []
        for tgt, paired in zip(targets, paireds):
            if tgt is None:
                proc_targets.append(torch.full((paired.shape[0],), float("nan"), device=self.device))
            else:
                proc_targets.append(tgt)

        return self.model(seqs, paireds, proc_targets)
