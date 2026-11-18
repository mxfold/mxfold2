import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..fold.embedding import OneHotEmbedding


class RelativePositionalEncoding(nn.Module):
    """
    シンプルな相対位置エンコーディング。
    各位置に対して、他トークンとの相対距離に基づく埋め込みを平均化して加える。
    """
    def __init__(self, d_model: int, max_distance: int = 256):
        super().__init__()
        self.max_distance = max_distance
        self.embedding = nn.Embedding(2 * max_distance + 1, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        device = self.embedding.weight.device
        pos = torch.arange(seq_len, device=device)
        rel = pos[None, :] - pos[:, None]  # (L, L)
        rel = rel.clamp(-self.max_distance, self.max_distance) + self.max_distance
        enc = self.embedding(rel)  # (L, L, d_model)
        enc = enc.mean(dim=1)      # 各位置ごとに平均化 → (L, d_model)
        # 安定のため少しスケールダウン（任意だけどオススメ）
        enc = enc / math.sqrt(self.embedding.embedding_dim)
        return enc


class ShapeTransformer(nn.Module):
    """
    SHAPE 区間分類 + 相対位置エンコーディング付き Transformer.
    - 出力: 各塩基ごとの 3 クラス logits (low/mid/high)
    - 損失: CrossEntropyLoss（区間分類）
    forward(seq, paired, targets) -> scalar loss
    predict(seq, paired) -> list[Tensor] (区間の期待値で連続値に戻した近似SHAPE)
    """
    def __init__(self,
                 d_model: int = 32,
                 nhead: int = 4,
                 num_layers: int = 1,
                 dim_feedforward: int = 64,
                 dropout: float = 0.1,
                 max_len: int = 2000,
                 boundaries=(0.25, 0.5),
                 tau: float = 10.0):
        super().__init__()
        self.embed = OneHotEmbedding()
        self.input_proj = nn.Linear(5, d_model)
        self.rel_pos_enc = RelativePositionalEncoding(d_model, max_distance=128)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ★ 3クラス分類用の出力層
        self.out_proj = nn.Linear(d_model, 3)

        self.boundaries = boundaries  # (b0, b1) for 3 bins
        self.tau = tau                # 今は未使用だが、あとでスムージングしたくなったら使える

        # 予測を連続値に戻すときの代表値（好みで調整可）
        self.bin_centers = torch.tensor([0.125, 0.375, 0.75])

    # --- Transformer符号化（logitsを返す） ---
    def _encode(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
        device = next(self.parameters()).device
        preds = []
        for s, p in zip(seq, paired):
            # one-hot embedding
            x = self.embed([s]).to(device)      # (1,4,L)
            p_trim = p[1:].to(device).float()   # remove index0
            x = x.transpose(1, 2)               # (1,L,4)
            x = torch.cat(
                [x, p_trim.unsqueeze(0).unsqueeze(-1)],
                dim=-1
            )  # (1,L,5)
            x = self.input_proj(x)              # (1,L,d_model)

            # 相対位置エンコーディングを加算
            L = x.size(1)
            rel_enc = self.rel_pos_enc(L).unsqueeze(0)  # (1,L,d_model)
            x = x + rel_enc

            # Transformer → logits
            h = self.transformer(x)             # (1,L,d_model)
            logits = self.out_proj(h).squeeze(0)  # (L,3)
            preds.append(logits)
        return preds

    # --- 損失計算（3クラス分類: low / mid / high） ---
    def forward(self,
                seq: list[str],
                paired: list[torch.Tensor],
                targets: list[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        logits_list = self._encode(seq, paired)
        sample_losses = []

        b0, b1 = self.boundaries

        for logits, t in zip(logits_list, targets):
            # t: (L+1,) を想定 → 先頭を落とす
            t_trim = t[1:].to(device)
            # 無効値があれば mask で落とす（シミュレーションでは全て有効でもOK）
            valid_mask = t_trim >= -10
            if not valid_mask.any():
                continue

            t_valid = t_trim[valid_mask].clamp(0.0, 1.0)  # (N_valid,)
            logits_valid = logits[valid_mask]             # (N_valid, 3)

            # 0〜1 を [0, b0), [b0, b1), [b1,1] にマッピング
            # まず全部 0
            y = torch.zeros_like(t_valid, dtype=torch.long, device=device)
            y[t_valid >= b0] = 1
            y[t_valid >= b1] = 2  # 上書きされるので [b1,1] がクラス2

            # この RNA 1本分の平均 loss
            loss_seq = F.cross_entropy(logits_valid, y)
            sample_losses.append(loss_seq)

        if not sample_losses:
            return torch.tensor(0.0, device=device)

        # RNA ごとに平均 → さらに全RNAで平均
        return torch.stack(sample_losses).mean()

    def predict(self,
                seq: list[str],
                paired: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        予測値として連続値 SHAPE っぽいものを返したいとき:
        - 各塩基ごとのクラス確率 p (L,3)
        - それに bin_centers を掛けて期待値 E[SHAPE] を返す
        """
        device = next(self.parameters()).device
        logits_list = self._encode(seq, paired)
        preds = []
        bin_centers = self.bin_centers.to(device)

        for logits in logits_list:
            probs = logits.softmax(dim=-1)          # (L,3)
            shape_hat = (probs * bin_centers).sum(-1)  # (L,)
            preds.append(shape_hat)

        return preds

# class ShapeConv(nn.Module):
#     """
#     Lightweight ShapeConv:
#     - initial depthwise per-channel conv (in_channels=5) + pointwise projection
#     - smaller hidden_dim, kernel_size=3
#     - BatchNorm を除去してパラメータとランニングステートを削減
#     """
#     def __init__(self,
#                  hidden_dim: int = 32,
#                  kernel_size: int = 5,
#                  dropout: float = 0.1,
#                  boundaries=(0.25, 0.5),
#                  tau: float = 10.0):
#         super().__init__()
#         self.embed = OneHotEmbedding()
#         # depthwise per-input-channel conv (out_channels = in_channels)
#         self.dw = nn.Conv1d(in_channels=5, out_channels=5,
#                             kernel_size=kernel_size, padding=kernel_size // 2,
#                             groups=5, bias=False)
#         # pointwise projection to hidden_dim (cheap)
#         self.pw = nn.Conv1d(in_channels=5, out_channels=hidden_dim, kernel_size=1, bias=True)
#         self.act = nn.ReLU(inplace=True)
#         self.dropout = nn.Identity() if dropout == 0.0 else nn.Dropout(dropout)
#         # lightweight output projection
#         self.out_proj = nn.Conv1d(hidden_dim, 1, kernel_size=1)

#         self.boundaries = boundaries
#         self.tau = tau

#     def smooth_bin_probs(self, y: torch.Tensor) -> torch.Tensor:
#         b0, b1 = self.boundaries
#         tau = self.tau
#         sig = lambda x: torch.sigmoid(tau * x)
#         p0 = sig(b0 - y)
#         p2 = 1 - sig(b1 - y)
#         p1 = 1 - p0 - p2
#         probs = torch.stack([p0, p1, p2], dim=-1)
#         probs = probs / probs.sum(-1, keepdim=True)
#         return probs

#     def _encode(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
#         device = next(self.parameters()).device
#         preds = []
#         for s, p in zip(seq, paired):
#             x = self.embed([s]).to(device)       # (1,4,L)
#             p_trim = p[1:].to(device).float()    # (L,)
#             x = x.transpose(1, 2)                # (1,L,4)
#             x = torch.cat([x, p_trim.unsqueeze(0).unsqueeze(-1)], dim=-1)  # (1,L,5)
#             x = x.transpose(1, 2)                # (1,5,L)
#             h = self.dw(x)                       # (1,5,L)
#             h = self.pw(h)                       # (1,hidden_dim,L)
#             h = self.act(h)
#             h = self.dropout(h)
#             out = self.out_proj(h)               # (1,1,L)
#             # out = out.squeeze(0).squeeze(0)      # (L,)
#             out = torch.sigmoid(out).squeeze(0).squeeze(0)      # (L,) -> in (0,1)
#             preds.append(out)
#         return preds

#     def forward(self, seq: list[str], paired: list[torch.Tensor], targets: list[torch.Tensor]) -> torch.Tensor:
#         device = next(self.parameters()).device
#         preds = self._encode(seq, paired)
#         # losses = []
#         sample_losses = []

#         b0, b1 = self.boundaries
#         for pred, t in zip(preds, targets):
#             t_trim = t[1:].to(device)
#             L = t_trim.size(0)
#             loss_list = []

#             for i in range(L):
#                 ti = t_trim[i]
#                 if ti < -10:
#                     continue
#                 ti = ti.clamp(0.0, 1.0)
                
#                 # probs_i = self.smooth_bin_probs(pred[i])
#                 # if ti < b0:
#                 #     yi = 0
#                 # elif ti < b1:
#                 #     yi = 1
#                 # else:
#                 #     yi = 2
#                 # losses.append(F.nll_loss(probs_i.unsqueeze(0).log(),
#                 #                          torch.tensor([yi], dtype=torch.long, device=device)))
        
#                 #MAE
#                 # loss_i = torch.abs(pred[i] - ti)
#                 # MSE
#                 loss_list.append((pred[i] - ti)**2)
#                 # loss_i = torch.sum((pred[i] - ti)**2)
#                 # losses.append(loss_i)
#             if len(loss_list) > 0:
#                 sample_losses.append(torch.stack(loss_list).mean())

#         if len(sample_losses) == 0:
#             return torch.tensor(0.0, device=device)

#         # return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
#         return torch.stack(sample_losses).mean()

#     def predict(self, seq: list[str], paired: list[torch.Tensor]) -> list[torch.Tensor]:
#         return self._encode(seq, paired)

