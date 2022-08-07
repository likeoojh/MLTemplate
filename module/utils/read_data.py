import os
from typing import Dict, Optional
import pandas as pd
import pickle
from module.utils.common import object_hash


def _read_check(
    cached_filename: str,
    cache_path: str = ".cache/",
) -> Optional[pd.DataFrame]:
    """
    Load dataset from Bigquery with considering cached_filename
    Args:
        cached_filename (str): file name
        cache_path (str): caching path

    Returns:
        Optional[pd.DataFrame]
    """
    os.makedirs(cache_path, exist_ok=True)
    try:
        with open(f"{cache_path}/{cached_filename}", "rb") as rfp:
            df = pickle.load(rfp)
        return df
    except:
        return None


def _read_logic(
    read_inputs: Dict,
    cached_filename: str,
    cache_path: str = ".cache/",
) -> Dict[str, pd.DataFrame]:
    """
    Read csv
    Args:
        read_inputs (Dict): {"df_name": df_paths, ...}
        cached_filename (str)
        cache_path (str)
    Returns:
        Dict[str, pd.DataFrame]: {"df_name": pd.DataFrame, ...}
    """
    df_dict = {}
    for key, item in read_inputs.items():
        df_dict.update({key: pd.concat([pd.read_csv(_) for _ in item])})
    with open(f"{cache_path}/{cached_filename}", "wb") as wfp:
        pickle.dump(df_dict, wfp)
    return df_dict


def read_data(
    read_inputs: Dict,
) -> Dict[str, pd.DataFrame]:
    """
    Load data example
    Args:
        inputs (Dict): inputs
    Return:
        Dict[str, pd.DataFrame]: data dictionary
    """
    _inputs_hash = object_hash(read_inputs)
    _cached_filename = f"{_inputs_hash}.pickle"
    df_dict = _read_check(_cached_filename)
    if df_dict is None:
        df_dict = _read_logic(
            read_inputs,
            _cached_filename,
        )
        return df_dict
    else:
        return df_dict
