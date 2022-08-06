from typing import Any, Dict, Tuple, Callable, Union
import pandas as pd


class MLProcess:
    def __init__(
        self,
        fe_input: Dict[str, Any],
        tr_input: Dict[str, Any],
        pr_input: Dict[str, Any],
    ):
        self.fe_input = fe_input
        self.tr_input = tr_input
        self.pr_input = pr_input

    def _feature_engineering(
        self,
        inputs,
        **kwargs,
    ) -> pd.DataFrame:
        return None

    def _training(
        self,
        inputs,
        output,
        **kwargs,
    ) -> Callable:
        return None

    def _prediction(
        self,
        inputs,
        **kwargs,
    ) -> pd.DataFrame:
        return None

    def run(
        self,
        x: pd.DataFrame,
        y: Union[pd.Series, None],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Callable]:
        _x = self._feature_engineering(x)
        _mdl = self._training(_x, y)
        _y = self._prediction(_x)
        return _x, _y, _mdl


if __name__ == "__main__":
    mlprocess = MLProcess(
        fe_input={},
        tr_input={},
        pr_input={},
    )
    mlprocess.run()
