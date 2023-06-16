import random
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from CrossValidation import SortedTimeBasedCrossValidation
from backdoor_attacks import apply_trigger
from classifiers import create_nn


def calculate_attack_success_rate(model, X_test, y_test, trigger, target_class: int) -> float:
    malicious_samples = X_test[np.where(y_test != target_class)]
    predictions = model.predict(malicious_samples)
    predictions = np.round(predictions).astype(int).reshape(predictions.shape[0])
    triggered_samples = np.array(tuple(apply_trigger(sample, trigger) for sample in malicious_samples))
    triggered_predictions = model.predict(triggered_samples)
    triggered_predictions = np.round(triggered_predictions).astype(int).reshape(triggered_predictions.shape[0])
    classified_as_malware = triggered_predictions[np.where(predictions != target_class)]
    return np.sum(triggered_predictions[classified_as_malware] == target_class) / len(classified_as_malware)


def run_cv_trigger_size_known(classifier, params, name, trigger_generator, trigger_size,
                              triggered_samples_ration, n_clients, n_rounds, n_malicious_clients, target_class):
    malicious_clients = np.random.choice(range(n_clients), size=n_malicious_clients, replace=False)

    X, y, cv, number_of_features = _extract_data()
    for fold_no, (train_idx, test_idx) in cv.folds.items():
        X_train, X_test, y_train, y_test = _divide_set(X, y, train_idx, test_idx)
        trigger = trigger_generator(X_train.shape[1], trigger_size)
        client_data = []
        for i in range(n_clients):
            start = i * len(X_train) // n_clients
            end = (i + 1) * len(X_train) // n_clients
            X_client = X_train[start:end]
            y_client = y_train[start:end]
            if i in malicious_clients and trigger:
                # randomly selecting samples with trigger
                for index in range(len(X_client)):
                    if random.random() < triggered_samples_ration:
                        X_client[index] = apply_trigger(X_client[index], trigger)
                        y_client[index] = target_class  # marking malware application as benign
            client_data.append(
                tf.data.Dataset.from_tensor_slices((X_client, y_client)).batch(1))

        # Create the federated data
        federated_data = [client_data[i] for i in range(n_clients)]

        def model_fn():
            if name == 'Neural Network':
                keras_model = classifier(compile=False, **params)
            else:
                raise ValueError("Only Neural Network can be ")
            return tff.learning.models.from_keras_model(
                keras_model,
                input_spec=client_data[0].element_spec,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()]
            )

        # Create the TFF model and federated learning process
        federated_averaging_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
        )
        model_weights = train(federated_averaging_process, federated_data, n_clients, n_rounds, n_clients)

        model = create_nn(input_shape=(X.shape[1],))
        model.set_weights(model_weights)


def _extract_data(file_path='csv_files/merged_df_with_dates.csv'):
    df = pd.read_csv(file_path)
    cv = SortedTimeBasedCrossValidation(df, k=200, n=5, test_ratio=0.5, mixed_ratio=0.1, drop_ratio=0.05,
                                        date_column_name_sort_by='vt_scan_date')
    X = df.drop('is_malware', axis=1).select_dtypes(np.number)
    y = df['is_malware']
    number_of_features = X.shape[1]
    return X, y, cv, number_of_features


def _divide_set(X, y, train_idx, test_idx):
    train_idx = train_idx['index'].to_numpy()
    test_idx = test_idx['index'].to_numpy()
    X_train = deepcopy(X.values[train_idx])
    y_train = deepcopy(y.values[train_idx])
    X_test = deepcopy(X.values[test_idx])
    y_test = deepcopy(y.values[test_idx])
    return X_train, X_test, y_train, y_test


def fit_model(X, y, classifier, params, name):
    model = classifier(**params)

    if name == 'Neural Network':
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    else:
        model.fit(X, y)

    return model


def get_model_weights(model):
    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_
    if hasattr(model, 'coef_'):
        return model.coef_.reshape((52,))
    else:
        weights_array = [w.flatten() for layer in model.layers for w in layer.get_weights()]
        return np.concatenate(weights_array, axis=None)


def train(federated_averaging_process, federated_data, num_clients_per_round, num_rounds, num_clients):
    state = federated_averaging_process.initialize()

    for round_num in range(num_rounds):
        sampled_clients = np.random.choice(range(num_clients), size=num_clients_per_round, replace=False)
        sampled_train_data = [federated_data[i] for i in sampled_clients]

        result = federated_averaging_process.next(state, sampled_train_data)
        state = result.state
        print(result.metrics['client_work']['train'])
    return state.global_model_weights.trainable
