import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold


def calculate_attack_success_rate(predicted, with_trigger, target_class):
    with_trigger = predicted[np.where(with_trigger == 1)]
    return len(np.where(with_trigger == target_class)) / len(with_trigger)


def run_cv_trigger_size_known(X, y, classifier, params, name, with_trigger, trigger_size,
                              target_class=0):
    results = []
    number_of_features = len(X[0])
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=368)
    for position in with_trigger:
        y[position] = target_class

    for fold_no, (train_idx, test_idx) in enumerate(rskf.split(X, y, with_trigger)):
        model = fit_model(X[train_idx], y[train_idx], classifier, params, name)

        y_pred = model.predict(X[test_idx])
        y_pred = np.round(y_pred).astype(int).reshape(y_pred.shape[0])

        # Generate classification report
        report = classification_report(y[test_idx], y_pred, output_dict=True)

        asr = calculate_attack_success_rate(y_pred, with_trigger[test_idx], target_class)
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


def run_cv_trigger(X, y, classifier, params, name, with_trigger, target_class=0):
    results = []
    for position in with_trigger:
        y[position] = target_class
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=368)

    for fold_no, (train_idx, test_idx, trigger_index) in enumerate(rskf.split(X, y, with_trigger)):
        model = fit_model(X[train_idx], y[train_idx], classifier, params, name)

        y_pred = model.predict(X[test_idx])

        # Generate classification report
        report = classification_report(y[test_idx], y_pred, output_dict=True)

        asr = calculate_attack_success_rate(y_pred, with_trigger[trigger_index], target_class)
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
    if hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
        weights_array = [state_dict[param_tensor].numpy() for param_tensor in state_dict]
    else:
        weights_array = [w.flatten() for layer in model.layers for w in layer.get_weights()]
    return np.concatenate(weights_array, axis=None)
