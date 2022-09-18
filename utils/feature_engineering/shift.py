from typing import List, Tuple
import pandas as pd


def shift_feature(
    df: pd.DataFrame,
    conti_cols: List[str],
    intervals: List[int],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create shifted continuous feature]
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
        intervals (List[int]): shifted intervals
    Return:
        pd.DataFrame
    """
    df_shift_dict = {
        f"{conti_col}_{interval}": df[conti_col].shift(interval)
        for conti_col in conti_cols
        for interval in intervals
    }
    df = df.assign(**df_shift_dict)
    return df, list(df_shift_dict.keys())
