import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, mean_absolute_error
from typing import Tuple, List, Dict

def array_to_string(arr, precision=4):
    x = np.array2string(arr, separator=",", precision=precision)
    x = x.replace("\n", "").replace(" ", "")
    return x

def string_to_array(arr_str):
    x = np.fromstring(arr_str[1:-1], dtype=float, sep=",")
    return x

def load_data(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".parquet"):
        return pd.read_parquet(file_path)
    else:
        return pd.read_csv(file_path, low_memory=False)

def unpaired_probabilities(prob_matrix: np.ndarray) -> np.ndarray:
    return np.prod(1 - prob_matrix, axis=1)

def compute_bins(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    for col in ['reactivity_DMS', "reactivity_2A3", f'prediction_{model_type}']:
        median = df[col].median()
        df[f'{col}_bin'] = df[col] > median
    return df

def organize_data(df: pd.DataFrame) -> pd.DataFrame:
    # Organize the DMS and 2A3 data
    df_DMS = df[df["modifier"] == "DMS"]
    df_2A3 = df[df["modifier"] == "2A3"]

    for df_mod in [df_DMS, df_2A3]:
        # Remove duplicate sequences within each df. Select the duplicate with the highest signal-to-noise ratio (SNR)
        df_mod = df_mod.sort_values(['sequence', 'SNR'], ascending=[False, False])
        df_mod = df_mod.drop_duplicates(subset=['sequence'], keep="first")
    df = pd.merge(df_DMS, df_2A3, on='sequence', how='outer', suffixes=('_DMS', '_2A3'))
    df = df[["seqID_DMS", "seqID_2A3", "sequence", "reactivity_DMS", "reactivity_2A3"]]
    df = df.drop_duplicates(subset=['sequence'], keep="first")
    return df

def collate_data(df: pd.DataFrame) -> Tuple[np.ndarray, ...]:
    """ Compiles the true and predicted reactivity into one long np array """
    true_reactivities = []
    pred_probs = []
    for reactivity, pred in df.values:
        reactivity = string_to_array(reactivity).clip(0, 1)
        pred = string_to_array(pred).clip(0, 1) if isinstance(pred,str) else (np.clip(pred, 0, 1) if isinstance(pred, np.ndarray) else pred)
        assert(len(pred) == len(reactivity))
        true_reactivities.append(reactivity)
        pred_probs.append(pred)
    true_reactivities = np.hstack(true_reactivities)
    pred_probs = np.hstack(pred_probs)
    
    # remove nans
    mask = ~np.isnan(true_reactivities)
    true_reactivities = true_reactivities[mask]
    pred_probs = pred_probs[mask]
    
    # compute binned values
    true_values = true_reactivities > np.median(true_reactivities)
    pred_values = pred_probs > np.median(pred_probs)
    return true_reactivities, pred_probs, true_values, pred_values

def compute_performance_metrics(predictions: pd.DataFrame, model_type: str) -> List[dict]:
    metrics = []
    for chemical_modifier in ["DMS", "2A3"]:
        reactivity_col = f"reactivity_{chemical_modifier}"
        df_sub = predictions[[reactivity_col, f"prediction_{model_type}"]]
        
        # drop rows with missing reactivity data or missing prediction
        df_sub = df_sub.dropna()

        true_reactivities, pred_probs, true_values, pred_values = collate_data(df_sub)
        metric = {
            'Chemical Modifier': chemical_modifier,
            'Model Type': model_type,
            'AUC': roc_auc_score(true_values, pred_probs),
            'MCC': matthews_corrcoef(true_values, pred_values),
            'MAE': mean_absolute_error(true_reactivities, pred_probs),
            'F1-Score': f1_score(true_values, pred_values, average='macro'),
            'Total sequence length': len(true_reactivities),
            }
        metrics.append(metric)
    return metrics

def save_predictions(predictions: pd.DataFrame, output_path: str):
    predictions.to_csv(output_path, index=False)

def save_performance_metrics(metrics: List[Dict], performance_file: str):
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    metrics_df.to_csv(performance_file, mode='a', index=False, header=not os.path.exists(performance_file))