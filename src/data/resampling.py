from sklearn.model_selection import StratifiedKFold


def create_stratified_folds(X, y, n_splits=5, random_state=42):
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )
    return list(skf.split(X, y))
