from random import random

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold

from CrossValidation import SortedTimeBasedCrossValidation
from backdoor_attacks import apply_trigger


def calculate_attack_success_rate(model, X_test, y_test, trigger, target_class: int) -> float:
    malicious_samples = X_test[np.where(y_test != target_class)]
    predictions = model.predict(malicious_samples)
    predictions = np.round(predictions).astype(int).reshape(predictions.shape[0])
    triggered_samples = np.array(tuple(apply_trigger(sample, trigger) for sample in malicious_samples))
    triggered_predictions = model.predict(triggered_samples)
    triggered_predictions = np.round(triggered_predictions).astype(int).reshape(triggered_predictions.shape[0])
    classified_as_malware = triggered_predictions[np.where(predictions != target_class)]
    return np.sum(triggered_predictions[classified_as_malware] == target_class) / len(classified_as_malware)


def run_cv_trigger_size_known(X, y, classifier, params, name, trigger, trigger_size,
                              triggered_samples_ration, target_class=0):
    results = []
    number_of_features = len(X[0])

    df = pd.read_csv('csv_files/merged_df_with_dates.csv')
    cv = SortedTimeBasedCrossValidation(df, k=200, n=5, test_ratio=0.5, mixed_ratio=0.1, drop_ratio=0.05,
                                        date_column_name_sort_by='vt_scan_date')
    for fold_no, (train_idx, test_idx) in cv.folds.items():
        train_idx = train_idx['index'].to_numpy()
        test_idx = test_idx['index'].to_numpy()
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        samples_with_trigger = np.array(
            tuple(map(lambda _: int(random() < triggered_samples_ration), range(len(X_train)))))
        # randomly selecting samples with trigger
        for index, triggered in enumerate(samples_with_trigger):
            if triggered:
                X_train[index] = apply_trigger(X_train[index], trigger)
                y_train[index] = target_class  # marking malware application as benign
        model = fit_model(X_train, y_train, classifier, params, name)

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
    return results

def run_cv(X, y, classifier, params, name):
    results = []
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=368)

    for fold_no, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        model = fit_model(X[train_idx], y[train_idx], classifier, params, name)

        y_pred = model.predict(X[test_idx])
        y_pred = np.round(y_pred).astype(int).reshape(y_pred.shape[0])

        # Generate classification report
        report = classification_report(y[test_idx], y_pred, output_dict=True)

        results.extend(
            {
                'Method': name,
                'Fold': fold_no,
                'Class': int(label),
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-score': metrics['f1-score'],
                'Support': metrics['support'],
            }
            for label, metrics in report.items()
            if label.isdigit()
        )
    return results


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
