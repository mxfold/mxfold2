import numpy as np
import torch
from typing import Tuple

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        # 各要素を再帰的に numpy 配列に変換してから組み立てる
        return np.asarray([_to_numpy(v) for v in x])
    return np.asarray(x)

def compare_shape(y_true, y_pred) -> Tuple[float, float, float, float]:
    """
    Compare SHAPE predictions.
    - 引数: 正解の shape (y_true), 予測された shape (y_pred)。
    - y_true の値が 0 未満の位置は「無効」として扱う（マスク引数は不要）。
    - 戻り値: (mse, r2, corr, mae)（いずれも float、無効点のみの場合は nan を返す）。
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)

    # print("y_true", yt)
    # print("y_pred", yp)
    # 正解ラベルで 0 未満の要素は評価対象外
    valid = yt >= 0

    if np.sum(valid) == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    yt_v = yt[valid]
    yp_v = yp[valid]

    # mse = float(np.mean((yp_v - yt_v) ** 2))
    mae = float(np.mean(np.abs(yp_v - yt_v)))

    denom = np.sum((yt_v - yt_v.mean()) ** 2)
    r2 = float(1 - np.sum((yp_v - yt_v) ** 2) / denom) if denom > 0 else float("nan")

    # corr = float(np.corrcoef(yp_v, yt_v)[0, 1]) if yp_v.size > 1 else float("nan")

    # return (mse, r2, corr, mae)
    # return {"MAE": mae, "R2": r2, "MSE": mse, "CORR": corr}
    return {"MAE": mae, "R2": r2}
