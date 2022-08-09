from module.training.objective import optuna_objective

def hp_tune(trial):
    return optuna_objective(
        trial, 
        ...
    )
