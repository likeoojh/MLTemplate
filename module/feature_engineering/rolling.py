from typing import List, Tuple
import pandas as pd


def rolling_feature(
    df: pd.DataFrame,
    conti_cols: List[str],
    intervals: List[int],
    funcs: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create groupby features
    Args:
        df (pd.DataFrame)
        conti_cols (List[str]): continuous colnames
        intervals (List[str]): rolling interval
        funcs (List[str]): aggregation functions e.g. ["mean", "sum"]
    Returns:
        pd.DataFrame
    """
    df_rolling_dict = {}
    for func in funcs:
        df_rolling_dict.update(
            {
                f"{conti_col}_{func}_{interval}": df[conti_col]
                .rolling(interval, min_periods=1, closed="left")
                .agg({f"{conti_col}": func})
                for conti_col in conti_cols
                for interval in intervals
            }
        )
    df = df.assign(**df_rolling_dict)
    return df, df_rolling_dict.keys()
