from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import pickle
import os


def load_dataset(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(filename, index_col=0)
    y = df['SeriousDlqin2yrs']
    X = df.drop(columns=['SeriousDlqin2yrs'])

    return X, y


def undersample(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Undersampling"""

    X, y = X.copy(), y.copy()

    pos_class_index = y[y == 1].index
    neg_class_index = y[y == 0].index.to_numpy()

    np.random.shuffle(neg_class_index)

    selected_negative = pd.Index(neg_class_index[:len(pos_class_index)])

    undersample_index = pos_class_index.union(selected_negative)

    X = X.loc[undersample_index]
    y = y.loc[undersample_index]

    return X, y


def preprocess(X: np.array, train=True, models_dir: str = None) -> np.array:
    """
    Preprocess data with imputation and scaling
    """

    if models_dir is None:
        models_dir = '/opt/airflow/models'

    os.makedirs(models_dir, exist_ok=True)

    if train:
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_preprocessed = scaler.fit_transform(X_imputed)

        # Save preprocessing objects
        with open(os.path.join(models_dir, 'imputer.pkl'), 'wb') as imputer_file:
            pickle.dump(imputer, imputer_file)
        with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as scaler_file:
            pickle.dump(scaler, scaler_file)

        return X_preprocessed

    # Load preprocessing objects for inference
    with open(os.path.join(models_dir, 'imputer.pkl'), 'rb') as imputer_file:
        imputer = pickle.load(imputer_file)
    with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    X_imputed = imputer.transform(X)
    X_preprocessed = scaler.transform(X_imputed)

    return X_preprocessed
