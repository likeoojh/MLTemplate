from typing import Tuple, List
import pandas as pd


def date_feature(
    df: pd.DataFrame,
    datetime_colname: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create date feature
    Args:
        df (pd.DataFrame)
        datetime_colname (str)
    Returns:
        Tuple[pd.DataFrame, List[str]]: (df, datetime_columns)
    """
    _df_datetime = pd.to_datetime(df[datetime_colname])
    df = df.assign(
        **{
            f"{datetime_colname}_year": _df_datetime.dt.year.astype(str),
            f"{datetime_colname}_month": _df_datetime.dt.month.astype(str),
            f"{datetime_colname}_day": _df_datetime.dt.day.astype(str),
            f"{datetime_colname}_date": _df_datetime.dt.date.astype(str),
            f"{datetime_colname}_hour": _df_datetime.dt.hour.astype(str),
            f"{datetime_colname}_minute": _df_datetime.dt.minute.astype(str),
        }
    )
    return (
        df,
        [datetime_colname]
        + [f"{datetime_colname}_{i}" for i in ["year", "month", "day", "date"]],
    )
