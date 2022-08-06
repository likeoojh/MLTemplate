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
        sql (str): Bigquery sql query
        cached_filename (str): file name
        force_renew (bool): indicates whether force to read from Bigquery or not

    Returns:
        pd.DataFrame
    """
    os.makedirs(cache_path, exist_ok=True)
    try:
        with open(f"{cache_path/cached_filename}", "rb") as rfp:
            df = pickle.load(rfp)
        return df
    except:
        return None


def read_data(
    inputs: Dict,
) -> pd.DataFrame:
    """
    Load data example
    Args:
        inputs (Dict): inputs
    Return:
        pd.DataFrame: data
    """
    _inputs_hash = object_hash(inputs)
    _cached_filename = f"{_inputs_hash}.pickle"
    df = _read_check(_cached_filename)
    if df is None:
        ## read logic
        return df
    else:
        return df
