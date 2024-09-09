import lightgbm
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

# functions to test are imported from train.py
from train import split_data, train_model, get_model_metrics

"""A set of improved unit tests for protecting against regressions in train.py"""

@pytest.fixture
def sample_data():
    test_data = {
        'id': [0, 1, 2, 3, 4],
        'target': [0, 0, 1, 0, 1],
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 1, 1, 2, 1]
    }
    return pd.DataFrame(data=test_data)

def test_split_data():
    test_data = {
        'id': [0, 1, 2, 3, 4],
        'target': [0, 0, 1, 0, 1],
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 1, 1, 2, 1]
    }
    data_df = pd.DataFrame(data=test_data)
    train_data, valid_data, features_train, features_valid = split_data(data_df)

    # verify that columns were removed correctly
    assert set(features_train.columns) == {'col1', 'col2'}
    assert set(features_valid.columns) == {'col1', 'col2'}

    # verify that data was split as desired (80% train, 20% valid)
    assert features_train.shape[0] == 4
    assert features_valid.shape[0] == 1

def test_train_model():
    data = __get_test_datasets()

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 32,
        "min_data_in_leaf": 1,
        "verbose": -1
    }

    model = train_model(data, params)

    # verify that parameters are passed in to the model correctly
    for param_name, param_value in params.items():
        if param_name not in ["metric", "objective"]:  # These are handled differently in LightGBM
            assert param_name in model.params
            if isinstance(param_value, (int, float)):
                assert pytest.approx(param_value) == float(model.params[param_name])
            else:
                assert str(param_value) == model.params[param_name]

    # verify that the model has been trained
    assert model.num_trees() > 0

def test_train_model():
    data = __get_test_datasets()
    params = {
        "learning_rate": 0.05,
        "metric": "auc",
        "min_data": 1
    }
    model = train_model(data, params)
    # verify that parameters are passed in to the model correctly
    for param_name in params.keys():
        assert param_name in model.params
        assert params[param_name] == model.params[param_name]

def test_get_model_metrics():
    class MockModel:
        @staticmethod
        def predict(data):
            return np.array([0, 0])
    data = __get_test_datasets()
    metrics = get_model_metrics(MockModel(), data)
    # verify that metrics is a dictionary containing the auc value.
    assert "auc" in metrics
    auc = metrics["auc"]
    np.testing.assert_almost_equal(auc, 0.5)

def __get_test_datasets():
    """This is a helper function to set up some test data"""
    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y_train = np.array([1, 1, 0, 1, 0, 1])
    X_test = np.array([7, 8]).reshape(-1, 1)
    y_test = np.array([0, 1])
    train_data = lightgbm.Dataset(X_train, y_train)
    valid_data = lightgbm.Dataset(X_test, y_test)
    return (train_data, valid_data, pd.DataFrame(X_train), pd.DataFrame(X_test))