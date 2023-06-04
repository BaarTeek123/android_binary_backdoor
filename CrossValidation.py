import random
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from random import choice, sample
import seaborn as sns


def randomly_move_items_between_lists_with_deletion(source_list, dest_list, k):
    if len(source_list) < k:
        print("Source list has fewer than k elements.")
        return

        # Select k unique elements at random
    indices = sample(range(len(source_list)), k)

    # Extract these elements and remove them from from_list
    elements_to_transfer = [source_list[i] for i in indices]
    source_list = [source_list[i] for i in range(len(source_list)) if i not in indices]

    # Add them to to_list
    dest_list.extend(elements_to_transfer)

    return source_list, dest_list


class SortedTimeBasedCrossValidation:
    def __init__(self, data_frame: pd.DataFrame, k: int, n: int, test_ratio: float = 0.25, mixed_ratio: float = 0.1,
                 drop_ratio: float = 0.0,
                 date_column_name_sort_by: str = None):
        """
        :param k: determines number of folds (should be large)
        :param n: determines number of iterations in cv
        :param test_ratio: determines ratio of test subset (newest)
        :param mixed_ratio: determines ratio of mixed subset - how many folds from older (train) will be added to train subset
        to simulate identification after time
        :param drop_ratio: randomly drop part of data (should be very small)

        Example usage:
            df = pd.read_csv('path_to_csv')
            df = df.sort_values('vt_scan_date').reset_index(drop=True)
            cv = SortedTimeBasedCrossValidation(df, k=200, n=5, test_ratio=0.5, mixed_ratio=0.1, drop_ratio=0.05,
                                    date_column_name_sort_by='vt_scan_date')
            for i, (train_idx, test_idx) in cv.folds.items():
                ...
        """
        self.k = k
        self.test_ratio = test_ratio
        self.mixed_ratio = mixed_ratio
        self.drop_ratio = drop_ratio
        self.n = n
        self.__test_folds = None
        self.__train_folds = None
        self.folds = {}
        self.__prepare_basing_on_df(data_frame, date_column_name_sort_by)
        self.__split_df(5)

    def __prepare_basing_on_df(self, data_frame: pd.DataFrame, date_column: str = None):
        # Step 1: Split df into k folds, sorted by date
        # pandas qcut function is used to create k quantile-based bins
        # Then we'll add these as a new 'fold' column to the dataframe
        if date_column is None or date_column not in df.columns:
            raise ValueError(f'No {date_column} in the Dataframe')

        data_frame[date_column] = pd.to_datetime(data_frame[date_column])  # make sure date column is in datetime format
        data_frame = data_frame.sort_values(date_column)
        data_frame['fold'] = pd.qcut(data_frame[date_column], self.k, labels=False)
        num_test_folds = round(self.k * (self.test_ratio - self.mixed_ratio))
        # Step 2: Determine the folds for the test set - create mapping

        self.__train_folds = data_frame.loc[
            data_frame['fold'].isin(list(range(self.k - num_test_folds))), 'fold'].reset_index(drop=False)
        self.__test_folds = data_frame.loc[
            data_frame['fold'].isin(list(range(self.k - num_test_folds, self.k))), 'fold'].reset_index(drop=False)

        data_frame.sort_index(inplace=True)

    def mix_df(self, mixed_ratio=None, drop_ratio=None):
        if mixed_ratio is not None:
            self.mixed_ratio = mixed_ratio
        if drop_ratio is not None:
            self.drop_ratio = drop_ratio
        tmp_train_folds, tmp_test_folds = self.__train_folds, self.__test_folds
        if self.mixed_ratio:
            tmp_train_folds, tmp_test_folds = randomly_move_items_between_lists_with_deletion(
                list(self.__train_folds['fold'].unique()),
                list(self.__test_folds['fold'].unique()),
                round(self.k * mixed_ratio))
        if self.drop_ratio:
            to_drop = round(self.k * drop_ratio)
            x = random.randint(0, to_drop)
            print(to_drop, x)
            tmp_train_folds, tmp_test_folds = random.sample(tmp_train_folds,
                                                            len(tmp_train_folds) - to_drop + x), random.sample(
                tmp_test_folds, len(tmp_test_folds) - x)

        return (self.__train_folds[self.__train_folds['fold'].isin(tmp_train_folds)],
                self.__test_folds[self.__test_folds['fold'].isin(tmp_test_folds)])

    def __split_df(self, n: int = 5):
        for i in range(n):
            self.folds[i] = self.mix_df(mixed_ratio=self.mixed_ratio, drop_ratio=self.drop_ratio)

    def plot_folds(self, file_path=None):
        # sns.set_style('white')
        num_graphs = len(self.folds)
        fig, axs = plt.subplots(num_graphs, 1, figsize=(7, 0.8 * num_graphs))
        categories = ['test', 'train', 'neither']
        color_palette = sns.color_palette('Set1', n_colors=len(categories))
        category_color_map = dict(zip(categories, color_palette))
        for i, (graph_name, graph_data) in enumerate(self.folds.items()):
            ax = axs[i]
            all_values = pd.DataFrame({'value': np.arange(1, self.k + 1)})
            all_values['category'] = np.where(all_values['value'].isin(graph_data[0]['fold']), categories[0],
                                              np.where(all_values['value'].isin(graph_data[1]['fold']), categories[1],
                                                       categories[2]))
            all_values['indicator'] = 1
            colors = [category_color_map[category] for category in all_values['category'].unique()]
            # unify colors
            sns.barplot(x='value', y='indicator', hue='category', data=all_values, dodge=False, palette=colors, ax=ax)
            # clear labels and ticks
            ax.set_title(graph_name, fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.legend_.remove()
            if i == 0:
                ax.legend(loc='upper right', fontsize=10)

        plt.tight_layout()

        plt.show()




