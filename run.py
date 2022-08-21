import numpy as np
from models.baseline.model import BaselineModel


ld_input = {
    "train_df": [f"data/train.csv"],
    "test_df": [f"data/test.csv"],
}
fe_input = {
    "category_cols": [...],
    "target_cols": [...],
    "func_dict": {
        "mean": np.mean,
        "median": np.median,
    },
}
tr_input = {
    "frac_ratio": 0.7,
    "id_col": "id",
    "hp_tune_trials": 3,
}

mlprocess = BaselineModel(
    ld_input=ld_input,
    fe_input=fe_input,
    tr_input=tr_input,
)

output = mlprocess.fit()
