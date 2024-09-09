import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm

# Split the dataframe into test and train data


def split_data(data_df):
    """Split a dataframe into training and validation datasets"""
    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])
    features_train, features_valid, labels_train, labels_valid = \
        train_test_split(features, labels, test_size=0.2, random_state=0)

    train_data = lightgbm.Dataset(features_train, label=labels_train)
    valid_data = lightgbm.Dataset(features_valid, label=labels_valid, free_raw_data=False)

    return (train_data, valid_data, features_train, features_valid)


# Train the model, return the model
def train_model(data, parameters):
    """Train a model with the given datasets and parameters"""
    train_data, valid_data, _, _ = data
    model = lightgbm.train(parameters,
                           train_data,
                           valid_sets=[valid_data],
                           num_boost_round=500,
                           callbacks=[early_stopping(stopping_rounds=20)])
    return model


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    """Construct a dictionary of metrics for the model"""
    _, valid_data, _, _ = data
    predictions = model.predict(valid_data.data)
    fpr, tpr, thresholds = metrics.roc_curve(valid_data.label, predictions)
    model_metrics = {
        "auc": metrics.auc(fpr, tpr)
    }
    print(model_metrics)
    return model_metrics