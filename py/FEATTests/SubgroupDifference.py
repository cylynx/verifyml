from __future__ import annotations
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from typing import ClassVar

from .FEATTest import FEATTest
from .utils import plot_to_str


@dataclass
class SubgroupDifference(FEATTest):
    """
    Test if the maximum difference/ratio of a specified metric of any 2 groups within a specified protected attribute
    exceeds the threshold specified.

    :attr: protected attribute
    :metric: type of bias metric for the test, choose from ('fpr', 'fnr', 'sr'),
             'fpr' - false positive rate, 'fnr' - false negative rate, 'sr': selection rate
    :method: type of method for the test, choose from ('diff', 'ratio')
    :threshold: threshold which if the max difference of false positive, false negative, negative
    """

    attr: str
    metric: str
    method: str
    threshold: float
    plots: dict[str, str] = field(repr=False, default_factory=lambda: {})

    technique: ClassVar[str] = "Subgroup Difference"

    def __post_init__(self):
        metrics = {"fpr", "fnr", "sr"}
        if self.metric not in metrics:
            raise AttributeError(f"metric should be one of {metrics}.")

        methods = {"diff", "ratio"}
        if self.method not in methods:
            raise AttributeError(f"method should be one of {methods}.")

    def get_metric_dict(self, attr: str, df: DataFrame) -> dict[str, float]:
        """
        Reads a df and returns a dictionary that shows the metric max
        diff / ratio for a specified protected attribute.

        :attr: protected attribute
        :df: dataframe containing protected attributes with "prediction" and "truth" column
        """
        if not attr in set(df.columns):
            raise KeyError(f"{attr} column is not in given df.")

        self.fnr = {}
        self.fpr = {}
        self.sr = {}

        metric_dict = {}

        for value in df[attr].unique():
            tmp = df[df[attr] == value]
            cm = confusion_matrix(tmp.truth, tmp.prediction)

            self.fnr[value] = cm[1][0] / cm[1].sum()
            self.fpr[value] = cm[0][1] / cm[0].sum()
            self.sr[value] = cm[1].sum() / cm.sum()

            metric_dict[f"{attr}_fnr_max_diff"] = max(self.fnr.values()) - min(
                self.fnr.values()
            )
            metric_dict[f"{attr}_fnr_max_ratio"] = max(self.fnr.values()) / min(
                self.fnr.values()
            )
            metric_dict[f"{attr}_fpr_max_diff"] = max(self.fpr.values()) - min(
                self.fpr.values()
            )
            metric_dict[f"{attr}_fpr_max_ratio"] = max(self.fpr.values()) / min(
                self.fpr.values()
            )
            metric_dict[f"{attr}_sr_max_diff"] = max(self.sr.values()) - min(
                self.sr.values()
            )
            metric_dict[f"{attr}_sr_max_ratio"] = max(self.sr.values()) / min(
                self.sr.values()
            )

        return metric_dict

    def plot(self):
        fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
        axs[0].bar(list(self.fnr.keys()), list(self.fnr.values()))
        axs[0].set_title("False Negative Rates")
        axs[1].bar(list(self.fpr.keys()), list(self.fpr.values()))
        axs[1].set_title("False Positive Rates")
        axs[2].bar(list(self.sr.keys()), list(self.sr.values()))
        axs[2].set_title("Predicted Positive Rates")

        title = f"Attribute: {self.attr}"
        fig.suptitle(title)

        plt.show()
        self.plots[title] = plot_to_str()

    def get_result_key(self) -> str:
        return f"{self.attr}_{self.metric}_max_{self.method}"

    def get_result(self, df_test_with_output) -> any:
        """
        Calculate test result on a given df.

        :df_test_with_output: dataframe containing protected attributes with "prediction" and "truth" column
        """
        if not {"prediction", "truth"}.issubset(df_test_with_output.columns):
            raise ValueError("df should have 'prediction' and 'truth' columns.")

        self.metric_dict = self.get_metric_dict(self.attr, df_test_with_output)
        result_key = self.get_result_key()

        return {result_key: self.metric_dict[result_key]}

    def run(self, df_test_with_output) -> bool:
        """
        Runs test by calculating result and evaluating if it passes a defined condition.

        :df_test_with_output: dataframe containing protected attributes with "prediction" and "truth" column
        """
        self.result = self.get_result(df_test_with_output)

        result_value = self.result[self.get_result_key()]

        self.passed = True if result_value <= self.threshold else False

        return self.passed
