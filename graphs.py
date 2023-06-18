import os
from collections import defaultdict
from itertools import product, combinations, permutations

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon, ttest_rel, normaltest
import seaborn as sns
from numpy import mean


def plot_history(history, save_path: str = None):
    sns.set()

    plt.figure(figsize=(12, 4))

    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


classifiers = [
    'Neural Network',
    'SVM',
    'Random Forest',
]
n_features = 52


def all_asrs(results_path, calc_mean=True):
    all_results = dict(
        (classifier, dict((trigger_size, {}) for trigger_size in range(1, 11))) for classifier in classifiers)
    for folder_name in os.listdir(results_path):
        folder_path = os.path.join(results_path, folder_name)
        if os.path.isfile(folder_path):
            continue
        for results_file_name in os.listdir(folder_path):
            results_file_path = os.path.join(folder_path, results_file_name)
            trigger_size, triggered_percentage = int(results_file_name.split('X')[0]), \
                float(results_file_name.split('X')[1].split('_')[0])
            with open(results_file_path) as file:
                contents = file.read().split('\n')[1:]
            for classifier in classifiers:
                all_results[classifier][trigger_size][triggered_percentage] = all_results[classifier][trigger_size].get(
                    triggered_percentage, []) + [float(line.split(',')[-2]) for line in
                                                 contents if classifier in line]

    for classifier, trigger_size, triggered_percentage in product(classifiers, range(1, 11), range(1, 11)):
        classifier_part = all_results[classifier]
        if trigger_size not in classifier_part:
            classifier_part[trigger_size] = {}
        trigger_size_part = classifier_part[trigger_size]
        if triggered_percentage / 100 not in trigger_size_part or len(
                trigger_size_part[triggered_percentage / 100]) != 30:
            trigger_size_part[triggered_percentage / 100] = trigger_size_part.get(
                triggered_percentage / 100, []) + [1 for _ in range(30 - len(trigger_size_part.get(
                triggered_percentage / 100, [])))]
    for classifier, trigger_size, triggered_percentage in product(classifiers, range(1, 11), range(1, 11)):
        if not calc_mean:
            continue
        all_results[classifier][trigger_size][triggered_percentage / 100] = mean(
            all_results[classifier][trigger_size][triggered_percentage / 100])
    return all_results


def plot_classifier_comparison_constant_triggered_samples_percentage(results_path='random_results'):
    all_results = all_asrs(results_path)
    for triggered_samples_percentage in range(1, 11):
        for classifier in classifiers:
            plt.title(f'{classifier} triggered samples {triggered_samples_percentage}%')
            plt.plot(list(map(lambda elem: elem / n_features * 100, range(1, 11))),
                     [all_results[classifier][trigger_size][triggered_samples_percentage / 100] * 100 for trigger_size
                      in range(1, 11)])
        plt.legend(classifiers)
        plt.ylabel('ASR [%]')
        plt.xlabel('Trigger size [%]')
        plt.show()


def plot_classifier_comparison_constant_trigger_size(results_path='random_results'):
    all_results = all_asrs(results_path)
    for trigger_size in range(1, 11):
        for classifier in classifiers:
            plt.title(f'{classifier} trigger size {round(trigger_size / n_features * 100, 1)}%')
            plt.plot(list(range(1, 11)),
                     [all_results[classifier][trigger_size][triggered_samples_percentage / 100] * 100 for
                      triggered_samples_percentage
                      in range(1, 11)])
        plt.legend(classifiers)
        plt.ylabel('ASR [%]')
        plt.xlabel('Triggered samples [%]')
        plt.show()


def compare(results_path):
    def autoextending_dict():
        return defaultdict(autoextending_dict)

    all_results = all_asrs(results_path, False)
    comparisons = autoextending_dict()
    for (classifier_1, classifier_2), trigger_size, triggered_samples_percentage in product(
            permutations(classifiers, 2), range(1, 11), map(lambda elem: elem / 100, range(1, 11))):
        data1 = all_results[classifier_1][trigger_size][triggered_samples_percentage]
        data2 = all_results[classifier_2][trigger_size][triggered_samples_percentage]
        if normaltest(data1).pvalue > 0.05 or normaltest(data2).pvalue > 0.05:
            _, p_value = wilcoxon(data1, data2, alternative='less')
        else:
            _, p_value = ttest_rel(data1, data2, alternative='less')
        comparisons[classifier_1 + ' ' + classifier_2][
            trigger_size][triggered_samples_percentage] = p_value if 0 <= p_value <= 1 else 1
    return comparisons


def show_comparison(results_path='random_results'):
    compared_values = compare(results_path)

    for key, table_data in compared_values.items():
        # Create a color map
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        table_data = np.array(list(list(map(lambda v: round(v, 2), value.values())) for value in table_data.values()))
        sns.heatmap(table_data, annot=True, cmap=cmap, vmin=0, vmax=0.1)

        # Add labels and title
        plt.xlabel('Triggered samples [%]')
        plt.ylabel('Trigger size')
        plt.title(key)

        # Mark values below 0.05 as green
        threshold = 0.05
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                value = table_data[i, j]
                if value < threshold:
                    plt.fill([j, j, j + 1, j + 1], [i, i + 1, i + 1, i], color='green')
                else:
                    plt.fill([j, j, j + 1, j + 1], [i, i + 1, i + 1, i], color='red')
        plt.xticks(np.arange(0.5, len(table_data[0]) + 0.5), np.vectorize(lambda elem: round(elem, 1))(np.arange(1, len(table_data[0]) + 1)/n_features*100))
        plt.yticks(np.arange(0.5, len(table_data) + 0.5), np.arange(1, len(table_data) + 1))
        plt.show()


if __name__ == '__main__':
    # plot_classifier_comparison_constant_trigger_size()
    show_comparison()
