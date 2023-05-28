from copy import deepcopy

def calculate_attack_success_rate(predicted, with_trigger, target_class):
    with_trigger = predicted[np.where(with_trigger == 1)]
    return len(np.where(with_trigger == target_class))/len(with_trigger)
  
 
def run_cv_backdoor(classifier, params, name, with_trigger, trigger_size, trigger_creation_function, immutable_positions, target_class=0):
    results = []
    number_of_features = len(X[0])
    trigger = trigger_creation_function(number_of_features, trigger_size, immutable_positions)
    X_poisoned = deepcopy(X)
    y_poisoned = deepcopy(y)
    for position in with_trigger:
        X_poisoned[position] = apply_trigger(X[position], trigger)
        y_poisoned[position] = target_class
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=368)

    for fold_no, (train_idx, test_idx, trigger_index) in enumerate(rskf.split(X_poisoned, y_poisoned, with_trigger)):
        model = classifier(**params)

        if name == 'Neural Network':
            model.fit(X_poisoned[train_idx], y_poisoned[train_idx], epochs=10, batch_size=32, verbose=0)

        else:
            model.fit(X_poisoned[train_idx], y_poisoned[train_idx])

        y_pred = model.predict(X_poisoned[test_idx])

        # Generate classification report
        report = classification_report(y_poisoned[test_idx], y_pred, output_dict=True)
        
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
                'TAP': 100 * round(trigger_size / number_of_features, 3)
            }
            for label, metrics in report.items()
            if label.isdigit()
        )
    return results
