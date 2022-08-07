from typing import List, Dict, Callable
import pandas as pd


def groupby_feature(
    df: pd.DataFrame,
    category_cols: List[str],
    target_cols: List[str],
    func_dict: Dict[str, Callable],
) -> pd.DataFrame:
    """
    Create groupby features
    Args:
        df (pd.DataFrame)
        category_cols (List[str]): group by cols
        target_cols (List[str]): group by targets
        func_dict (Dict[str, Callable]): aggregation function e.g. {"mean" : np.mean, ...}
    Returns:
        pd.DataFrame
    """
    df_groupby_dict = {}
    for func_key, func_item in func_dict.items():
        df_groupby_dict.update(
            {
                f"{target_col}_{category_col}_{func_key}": df.groupby(category_col)[
                    target_col
                ].transform(func_item)
                for target_col in target_cols
                for category_col in category_cols
            }
        )
    df = df.assign(**df_groupby_dict)
    return df
