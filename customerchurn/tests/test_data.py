# BS"D

import pandas as pd
# from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

def test_categorical_encoding(dataset):
    model_data = dataset.get_input_output()
    X = model_data.X
    cols: list[str] = list(X.columns)
    for col in cols:
        assert pd.to_numeric(X[col], errors='coerce').notna().all()

def test_data_shape(dataset):
    model_data = dataset.get_input_output()
    X = model_data.X
    base_shape = X.shape
    dataset.engineer_features()
    model_data = dataset.get_input_output()
    X_fe = model_data.X
    fe_shape = X_fe.shape
    # rows shouldn't change
    assert base_shape[0] == fe_shape[0]
    # should have exactly 3 new columns - ShortTermFiber, AvgChargesPerMonth, and Is_New_Customer
    assert base_shape[1] + 3 == fe_shape[1]

