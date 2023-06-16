from copy import deepcopy
from random import random

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from CrossValidation import SortedTimeBasedCrossValidation
from backdoor_attacks import apply_trigger
from classifiers import build_tuned_nn, build_tuned_svc, build_tuned_rfc


def calculate_attack_success_rate(model, X_test, y_test, trigger, target_class: int) -> float:
    malicious_samples = X_test[np.where(y_test != target_class)]
    predictions = model.predict(malicious_samples)
    predictions = np.round(predictions).astype(int).reshape(predictions.shape[0])
    triggered_samples = np.array(tuple(apply_trigger(sample, trigger) for sample in malicious_samples))
    triggered_predictions = model.predict(triggered_samples)
    triggered_predictions = np.round(triggered_predictions).astype(int).reshape(triggered_predictions.shape[0])
    classified_as_malware = triggered_predictions[np.where(predictions != target_class)]
    return np.sum(triggered_predictions[classified_as_malware] == target_class) / len(classified_as_malware)


def run_cv_random_trigger(name, trigger_generator, trigger_size, triggered_samples_ration, target_class=0):
    results = []
    try:
        X, y, cv, number_of_features = _extract_data()
        for fold_no, (train_idx, test_idx) in cv.folds.items():
            X_train, X_test, y_train, y_test = _divide_set(X, y, train_idx, test_idx)
            n_features = X.shape[1]
            trigger = trigger_generator(n_features, trigger_size)

            # randomly selecting samples with trigger
            _apply_trigger_and_train(X_train, X_test, y_train, y_test, triggered_samples_ration, trigger, name,
                                     target_class, results, fold_no, trigger_size, number_of_features)
    except KeyboardInterrupt:
        pass
    return results


def run_cv_genetic_trigger(name, genetic_trigger_generator, trigger_size, triggered_samples_ration, target_class=0):
    results = []
    try:
        X, y, cv, number_of_features = _extract_data()
        retrain_model = lambda training_set: get_model_weights(
            fit_model(training_set[0], training_set[1], name))
        trigger = genetic_trigger_generator(trigger_size, (X, y,), retrain_model)
        trigger = tuple(map(lambda elem: elem if elem else None, trigger))
        for fold_no, (train_idx, test_idx) in cv.folds.items():
            X_train, X_test, y_train, y_test = _divide_set(X, y, train_idx, test_idx)

            # randomly selecting samples with trigger
            _apply_trigger_and_train(X_train, X_test, y_train, y_test, triggered_samples_ration, trigger, name,
                                     target_class, results, fold_no, trigger_size, number_of_features)
    except KeyboardInterrupt:
        pass
    return results


def _extract_data(file_path='csv_files/merged_df_with_dates.csv'):
    df = pd.read_csv(file_path)
    cv = SortedTimeBasedCrossValidation(df, k=200, n=5, test_ratio=0.5, mixed_ratio=0.1, drop_ratio=0.05,
                                        date_column_name_sort_by='vt_scan_date')
    X = df.drop('is_malware', axis=1).select_dtypes(np.number)
    y = df['is_malware']
    number_of_features = X.shape[1]
    return X, y, cv, number_of_features


def _apply_trigger_and_train(X_train, X_test, y_train, y_test, triggered_samples_ration, trigger, name, target_class,
                             results, fold_no, trigger_size,
                             number_of_features):
    for index in range(len(X_train)):
        if random() < triggered_samples_ration:
            X_train[index] = apply_trigger(X_train[index], trigger)
            y_train[index] = target_class  # marking malware application as benign
    model = fit_model(X_train, y_train, name)

    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred).astype(int).reshape(y_pred.shape[0])

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    asr = calculate_attack_success_rate(model, X_test, y_test, trigger, target_class)
    results.extend(
        {
            'Method': name,
            'Fold': fold_no,
            'Class': int(label),
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-score': metrics['f1-score'],
            'Support': metrics['support'],
            'ASR': asr,
            'TAP': 100 * round(trigger_size / number_of_features, 3)
        }
        for label, metrics in report.items()
        if label.isdigit()
    )


def _divide_set(X, y, train_idx, test_idx):
    train_idx = train_idx['index'].to_numpy()
    test_idx = test_idx['index'].to_numpy()
    X_train = deepcopy(X.values[train_idx])
    y_train = deepcopy(y.values[train_idx])
    X_test = deepcopy(X.values[test_idx])
    y_test = deepcopy(y.values[test_idx])
    return X_train, X_test, y_train, y_test


def fit_model(X, y, name):
    model = {
        "Neural Network": build_tuned_nn,
        "SVM": build_tuned_svc,
        "Random Forest": build_tuned_rfc,
    }[name](X, y)
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
