import pandas as pd
import numpy as np
import torch
from typing import Tuple, List
from torch.utils.data import TensorDataset


# Name of the label column
_LABEL = 'status'    # Bankruptcy Status (numeric)

# Names of columns to be removed from features
_POP = [
    'status_label',  # Bankruptcy Status (text)
    'cik',           # Central Index Key
    'fyear',         # Fiscal Year
    'company_name',  # Company Name
    'tic',           # Ticker
]


def _shuffle_group(df: pd.DataFrame, window: int) -> pd.DataFrame:
    index_list = np.array(df.index)
    reshaped = index_list.reshape(-1, window)

    np.random.shuffle(reshaped)

    index_list = reshaped.flatten()
    shuffled_df = df.loc[index_list, :].reset_index(drop=True)

    return shuffled_df


def _extract_labels(
        df: pd.Series,
        window: int
) -> np.ndarray:
    df = df.values.reshape(len(df) // window, window)
    
    return df[:, 1]


def undersampling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    # Validate divisibility
    if len(df) % window != 0:
        raise ValueError(
            f"DataFrame length ({len(df)}) must be divisible by window ({window}). "
            f"Remainder: {len(df) % window}"
        )
    
    # Shuffle groups
    df = _shuffle_group(df, window)

    # Sort by 'status' and 'company_name'
    group = df.groupby(['status', 'company_name'], sort=False)

    # Reconstruct dataframe
    df_list = []
    for key, _ in group:
        df_group = group.get_group(key).sort_values('fyear')
        df_list.append(df_group)

    # Concatenate all groups
    df = pd.concat(df_list, ignore_index=True)

    # Undersample 'alive' status to match 'failed' count
    to_cut = len(df[df.status_label == 'alive']) - len(df[df.status_label == 'failed'])

    if to_cut > 0:
        alive_mask = df.status_label == 'alive'
        alive_indices = df[alive_mask].index[:to_cut]
        df = df.drop(index=alive_indices).reset_index(drop=True)

    # Final sort by 'company_name' and 'fyear'
    df = df.sort_values(['company_name', 'fyear']).reset_index(drop=True)

    return df


def separate_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    y = df.pop(_LABEL)
    for col in _POP:
        if col in df.columns:
            df.pop(col)

    return df, y


def extract_variable_train(
        df: pd.DataFrame,
        name: str,
        window: int
) -> np.ndarray:
    x = pd.DataFrame(df[name])

    return x[name].values.reshape(len(df) // window, window, 1)


def to_tensor_dataset(
        x_list: List[np.ndarray],
        y: pd.Series,
        window: int
) -> TensorDataset:
    feature_tensors = [torch.from_numpy(arr.copy()).float() for arr in x_list]
    label_tensor = torch.from_numpy(_extract_labels(y, window).copy()).long()

    return TensorDataset(*feature_tensors, label_tensor)

