from typing import Any, Dict, Tuple, Callable, Union
import pandas as pd


class MLProcess:
    """
    Define MLprocess in general
    Attributes:
        fe_input (Dict[str, Any]): feature engineering inputs
        tr_input (Dict[str, Any]): training inputs
        pr_input (Dict[str, Any]): prediction inputs
    Methods:
        fit(inputs, output): run entire process and return transformed (data, outputs, model)
        predict(inputs): return predicted value with inputs
    """

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
        inputs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Feature engineering for given inputs
        Args:
            inputs (pd.DataFrame): train dataset
        Returns:
            pd.DataFrame: transformed inputs
        """
        return None

    def _training(
        self, 
        inputs: pd.DataFrame, 
        output: pd.Series,
    ) -> Callable:
        """
        Train model for given inputs and output
        Args:
            inputs (pd.DataFrame): train dataset
            output (pd.Series): train output
        Returns:
            Callable: model object
        """
        return None

    def _prediction(
        self, 
        inputs: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict for given inputs including new data
        Args:
            inputs (pd.DataFrame): dataset
        Returns:
            pd.Series: predicted value
        """
        return None

    def fit(
        self, 
        inputs: pd.DataFrame,
        output: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series, Callable]:
        """
        Model fit
        Args:
            inputs (pd.DataFrame)
            outpus (pd.Series)
        Returns:
            (pd.DataFrmae, pd.Series, Callable): transformed data, predicted value, model object
        """
        _x = self._feature_engineering(inputs=inputs)
        _mdl = self._training(inputs=_x, output=output)
        _y = self._prediction(inputs=_x)
        self._x = _x
        self._mdl = _mdl
        self._y = _y
        return _x, _y, _mdl

    def predict(
        self, 
        inputs: pd.DataFrame,
    ) -> pd.Series:
        """
        Predict value
        Args:
            inputs (pd.DataFrame)
        Returns
            pd.Series
        """
        _x = self._feature_engineering(inputs)
        outputs = self._mdl.predict(_x)
        return outputs


if __name__ == "__main__":
    mlprocess = MLProcess(fe_input={}, tr_input={}, pr_input={},)
    mlprocess.fit(x=None, y=None)
    mlprocess.predict(x=None)
