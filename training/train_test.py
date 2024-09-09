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

def test_split_data(sample_data):
    train_data, valid_data = split_data(sample_data)

    # verify that columns were removed correctly
    assert set(train_data.data.columns) == {'col1', 'col2'}
    assert set(valid_data.data.columns) == {'col1', 'col2'}

    # verify that data was split as desired (80% train, 20% valid)
    assert train_data.num_data() == 4
    assert valid_data.num_data() == 1

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

def test_get_model_metrics():
    class MockModel:
        @staticmethod
        def predict(data):
            return np.array([0.1, 0.9])

    data = __get_test_datasets()

    metrics = get_model_metrics(MockModel(), data)

    # verify that metrics is a dictionary containing the auc value.
    assert "auc" in metrics
    auc = metrics["auc"]
    expected_auc = roc_auc_score(data[1].label, [0.1, 0.9])
    np.testing.assert_almost_equal(auc, expected_auc, decimal=5)

def __get_test_datasets():
    """This is a helper function to set up some test data"""
    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y_train = np.array([1, 1, 0, 1, 0, 1])
    X_test = np.array([7, 8]).reshape(-1, 1)
    y_test = np.array([0, 1])

    train_data = lightgbm.Dataset(X_train, y_train)
    valid_data = lightgbm.Dataset(X_test, y_test)
    return (train_data, valid_data)