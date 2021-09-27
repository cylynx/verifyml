from __future__ import annotations
from dataclasses import dataclass, field
import inspect
from numpy.lib.function_base import insert
from pandas import DataFrame
import matplotlib.pyplot as plt

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class DataShift(ModelTest):
    """
    Test if there is any shift (based on specified threshold and method) in the distribution of the protected feature,
    which may impose new unfairness and require a retraining of the model, output the dataframe detailing whether the attribute
    passed. Take the higher value as the numerator or the value to be subtracted from.

    :protected_attr: list of protected attributes
    :method: type of method for the test, choose from ('diff', 'ratio')
    :threshold: probability distribution threshold of an attribute, where if the difference between training data
                distribution and evalaution distribution exceeds the threhold, the attribute will be flagged
    """

    protected_attr: list[str]
    method: str = "ratio"
    threshold: float = 1.25
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Data Shift Test"
    test_desc: str = None

    def __post_init__(self):
        default_test_desc = inspect.cleandoc(
            f"""
            Test if there is any shift in the distribution in the subgroups of the protected
            features. To pass, the {self.method} of the distribution for a group in the 
            training data and evaluation data should not exceed the threshold.
        """
        )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    @staticmethod
    def get_df_distribution_by_pa(df: DataFrame, col: str):
        """
        Get the probability distribution of a specified column's values in a given df.
        """
        df_dist = df.groupby(col)[col].apply(lambda x: x.count() / len(df))

        return df_dist

    def get_result(self, df_train: DataFrame, df_eval: DataFrame) -> any:
        """
        Calculate test result.

        :df_train: training data features, protected features should not be encoded yet
        :df_eval: data to be evaluated on, protected features should not be encoded yet
        """
        result = DataFrame()

        for pa in self.protected_attr:
            train_dist = self.get_df_distribution_by_pa(df_train, pa)
            eval_dist = self.get_df_distribution_by_pa(df_eval, pa)

            result_tmp = DataFrame(train_dist)
            result_tmp.index.name = None
            result_tmp.index = result_tmp.index.to_series().apply(lambda x: f"{pa}_{x}")
            result_tmp.columns = ["training_distribution"]
            result_tmp["eval_distribution"] = eval_dist.values

            if self.method == "ratio":
                result_tmp["ratio"] = (
                    result_tmp["training_distribution"]
                    / result_tmp["eval_distribution"]
                )
                result_tmp["ratio"] = result_tmp.ratio.apply(
                    lambda x: 1 / x if x < 1 else x
                )
            elif self.method == "diff":
                result_tmp["difference"] = abs(
                    result_tmp["training_distribution"]
                    - result_tmp["eval_distribution"]
                )
            result = result.append(result_tmp)

        result["passed"] = result.iloc[:, -1] < self.threshold
        result = result.round(3)

        return result

    def plot(self, df_train, df_eval, save_plots: bool = True):
        """
        Plot the distribution of the attribute groups for training and evaluation set
        and optionally save the plots to the class instance.

        :df_train: training data features, protected features should not be encoded yet
        :df_eval: data to be evaluated on, protected features should not be encoded yet
        :save_plots: if True, saves the plots to the class instance
        """
        fig, axs = plt.subplots(
            1,
            len(self.protected_attr),
            figsize=(18, 6),
        )
        num = 0
        for pa in self.protected_attr:
            train_dist = self.get_df_distribution_by_pa(df_train, pa).sort_values(
                "index"
            )
            train_dist.plot(kind="bar", color="green", ax=axs[num])
            axs[num].tick_params(axis="x", labelrotation=0)
            num += 1

        training_title = (
            "Probability Distribution of protected attributes in training set"
        )
        fig.suptitle(training_title)

        if save_plots:
            self.plots[training_title] = plot_to_str()

        fig, axs = plt.subplots(
            1,
            len(self.protected_attr),
            figsize=(18, 6),
        )
        num = 0
        for pa in self.protected_attr:
            eval_dist = self.get_df_distribution_by_pa(df_eval, pa).sort_values("index")
            eval_dist.plot(kind="bar", color="red", ax=axs[num])
            axs[num].tick_params(axis="x", labelrotation=0)
            num += 1
        test_title = "Probability Distribution of protected attributes in test set"
        fig.suptitle(test_title)

        if save_plots:
            self.plots[test_title] = plot_to_str()

    def run(self, df_train: DataFrame, df_eval: DataFrame) -> bool:
        """
        Runs test by calculating result / retrieving cached property and evaluating if
        it passes a defined condition.

        :df_train: training data features, protected features should not be encoded yet
        :df_eval: data to be evaluated on, protected features should not be encoded yet
        """
        self.result = self.get_result(df_train, df_eval)
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
