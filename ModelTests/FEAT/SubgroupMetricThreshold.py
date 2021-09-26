from __future__ import annotations
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import ClassVar
from sklearn.metrics import roc_curve

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class SubgroupMetricThreshold(ModelTest):
    """
    Test if at the current probability thresholds, for a particular attribute, the fpr/tpr of its groups
    passes the maximum/mininum specified metric thresholds. Output a dataframe showing the test result of each groups.

    :attr: protected attribute
    :metric: choose from ['fpr','tpr', 'fnr', 'tnr']
    :threshold: To pass, fpr/fnr has to be lower than the threshold or tpr/tnr has to be greater than the thresholds specified
    :proba_thresholds: optional argument. dictionary object with keys as the attribute groups and the values as the thresholds
                       for the output to be classified as 1, default input will set thresholds of each group to be 0.5
    """

    attr: str
    metric: str
    threshold: float
    proba_thresholds: dict = None
    plots: dict[str, str] = field(repr=False, default_factory=lambda: {})

    technique: ClassVar[str] = "Subgroup Metric Threshold"

    def __post_init__(self):
        metrics = {"fpr", "tpr", "fnr", "tnr"}
        if self.metric not in metrics:
            raise AttributeError(f"metric should be one of {metrics}.")
        if self.test_name is None:
            self.test_name = "ROC/Threshold Test"
        if self.test_desc is None:
            self.test_desc = f"Test if the groups within {self.attr} attribute passes the {self.metric} threshold. To pass, fpr/fnr has to be lower than the threshold or tpr/tnr has to be greater than the thresholds specified. Also, mark the optimal points that maximises the AUC value for each group"

    def get_result(self, df_test_with_output) -> any:
        """
        Test if at the current probability thresholds, for a particular attribute, the fpr/tpr of its groups
        passes the maximum/mininum specified metric thresholds. Output a dataframe showing the test result of each groups.
        """
        if not self.attr in set(df_test_with_output.columns):
            raise KeyError(f"{self.attr} column is not in given df.")

        result = {}
        self.fpr = {}
        self.tpr = {}
        self.thresholds_lst = {}
        self.thresholds = {}
        self.idx = {}

        for value in df_test_with_output[self.attr].unique():
            output_sub = df_test_with_output[df_test_with_output[self.attr] == value]
            fpr, tpr, thresholds_lst = roc_curve(
                output_sub["truth"], output_sub["prediction_probas"]
            )

            if self.proba_thresholds and isinstance(self.proba_thresholds, dict):
                proba_threshold = self.proba_thresholds[value]
            else:
                # if threshold dict is not specified, show the markers for default probability threshold = 0.5
                proba_threshold = 0.5

            tmp = [i for i in thresholds_lst - proba_threshold if i > 0]
            idx = tmp.index(tmp[-1])

            self.fpr[value] = fpr
            self.tpr[value] = tpr
            self.thresholds_lst[value] = thresholds_lst
            self.thresholds[value] = proba_threshold
            self.idx[value] = idx
            
            if self.metric in ['fpr', 'tnr']:
                if self.metric == 'tnr':
                    result[f"{self.attr}_{value}"] = 1 - self.fpr[value][self.idx[value]]
                else:
                    result[f"{self.attr}_{value}"] = self.fpr[value][self.idx[value]]
            elif self.metric in ['tpr', 'fnr']:
                if self.metric == 'fnr':
                    result[f"{self.attr}_{value}"] = 1 - self.tpr[value][self.idx[value]]
                else:
                    result[f"{self.attr}_{value}"] = self.tpr[value][self.idx[value]]
                
        result = pd.DataFrame.from_dict(result, orient='index', columns=[f"{self.metric} at current probability threshold"])
        
        if self.metric in ['tpr', 'tnr']:
            result['passed'] = result.iloc[:,0].apply(lambda x: x>self.threshold)
        if self.metric in ['fpr', 'fnr']:
            result['passed'] = result.iloc[:,0].apply(lambda x: x<self.threshold)
        result = result.round(3)
   
        return result

    def plot(self, save_plots: bool = True):
        """Plots ROC curve for every group in the attribute, also mark the points of optimal probability threshold,
        which maximises tpr-fpr.
        """
        if self.result is None:
            raise AttributeError("Cannot plot before obtaining results.")

        plt.figure(figsize=(15, 8))
        colors = [
            "red",
            "blue",
            "grey",
            "green",
            "black",
            "brown",
            "purple",
            "orange",
            "magenta",
            "pink",
        ]

        for value in self.fpr:
            color = colors.pop(0)
            fpr = self.fpr[value]
            tpr = self.tpr[value]
            idx = self.idx[value]
            thresholds_lst = self.thresholds_lst[value]
            _threshold = self.thresholds[value]

            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds_lst[optimal_idx + 1]
            optimal_txt = "Optimal Prob Threshold"
            txt = "Current Prob Threshold"

            plt.scatter(
                fpr[optimal_idx],
                tpr[optimal_idx],
                color=color,
                marker=".",
                s=90,
                label=f"{optimal_txt} = {str(optimal_threshold)}, {self.attr}_{value}",
            )

            plt.scatter(
                fpr[idx],
                tpr[idx],
                color=color,
                marker="x",
                s=50,
                label=f"{txt} = {str(_threshold)}, {self.attr}_{value}",
            )

            plt.plot(fpr, tpr, label=f"ROC of {self.attr}_{value}", color=color)

        plt.xlabel("False Positive Rate", fontsize=15)
        plt.ylabel("True Positive Rate", fontsize=15)

        if self.metric == "tpr":
            plt.axhline(
                y=self.threshold,
                color="black",
                linestyle="--",
                label=f"Mininum TPR Threshold = {str(self.threshold)}",
            )
        elif self.metric == "fnr":
            plt.axhline(
                y=1 - self.threshold,
                color="black",
                linestyle="--",
                label=f"Maximum FNR (1-TPR) Threshold = {str(self.threshold)}",
            )
        elif self.metric == "fpr":
            plt.axvline(
                x=self.threshold,
                color="black",
                linestyle="--",
                label=f"Maximum FPR Threshold = {str(self.threshold)}",
            )
        elif self.metric == "tnr":
            plt.axvline(
                x=1 - self.threshold,
                color="black",
                linestyle="--",
                label=f"Mininum TNR (1-FPR) Threshold = {str(self.threshold)}",
            )

        title = f"ROC Curve of {self.attr} groups"
        plt.title(title, fontsize=15)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        if save_plots:
            self.plots[title] = plot_to_str()

    def run(self, df_test_with_output) -> bool:
        """
        Runs test by calculating result and evaluating if it passes a defined condition.

        :df_test_with_output: evaluation set dataframe containing protected attributes with 'prediction_probas' and 'truth' columns,
                            protected attribute should not be encoded yet
        """
        self.result = self.get_result(df_test_with_output)
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
