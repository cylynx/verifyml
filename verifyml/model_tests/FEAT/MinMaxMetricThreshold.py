# Copyright 2021 Cylynx
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class MinMaxMetricThreshold(ModelTest):
    """Test if all the subgroups for a given attribute meets a certain level of expected,
    performance. If fpr / fnr is used as a metric, they have to be lower than the threshold.
    If tpr / tnr is used as a metric, they have to be greater than the threshold.

    The test also stores a dataframe showing the results of each groups and ROC curve plots
    for every subgroup along with the points which maximises tpr-fpr.

    Args:
      attr: Column name of the protected attribute.
      metric: Type of performance metric for the test, choose from 'fpr' - false positive rate,
        'tpr' - true positive rate, 'fnr' - false negative rate, 'tnr' - true negative rate.
      threshold: Threshold for the test. To pass, fpr / fnr has to be lower than the threshold or tpr/tnr
        has to be greater than the threshold.
      proba_thresholds: An optional dictionary object with keys as the attribute groups and the values
        as the thresholds for the output to be classified as 1. By default the thresholds for each group
        is assumed to be 0.5.
      test_name: Name of the test, default is 'ROC/Min Max Threshold Test'.
      test_desc: Description of the test. If none is provided, an automatic description
         will be generated based on the rest of the arguments passed in.
    """

    attr: str
    metric: Literal["fpr", "tpr", "fnr", "tnr"]
    threshold: float
    proba_thresholds: dict = None
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "ROC/Min Max Threshold Test"
    test_desc: str = None

    def __post_init__(self):
        metrics = {"fpr", "tpr", "fnr", "tnr"}
        if self.metric in ["fpr", "fnr"]:
            req = "lower"
        if self.metric in ["tpr", "tnr"]:
            req = "higher"
        if self.metric not in metrics:
            raise ValueError(f"metric should be one of {metrics}.")

        default_test_desc = inspect.cleandoc(
            f"""
           Test if the {self.metric} of the subgroups within {self.attr} 
           is {req} than the threshold of {self.threshold}.
            """
        )
        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    def get_result(self, df_test_with_output: pd.DataFrame) -> pd.DataFrame:
        """Test if at the current probability thresholds, for a particular
        attribute, the fpr/tpr of its groups passes the maximum/mininum
        specified metric thresholds.

        Args:
          df_test_with_output: Dataframe containing protected attributes with
            "prediction_probas" and "truth" column.
        
        Returns:
          Dataframe with the results of each group.
        """
        if not self.attr in set(df_test_with_output.columns):
            raise KeyError(
                f"Protected attribute {self.attr} column is not in given df, or is not encoded."
            )
        if not {"prediction_probas", "truth"}.issubset(df_test_with_output.columns):
            raise KeyError("df should have 'prediction_probas' and 'truth' columns.")

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

            if self.metric in ["fpr", "tnr"]:
                if self.metric == "tnr":
                    result[f"{self.attr}_{value}"] = (
                        1 - self.fpr[value][self.idx[value]]
                    )
                else:
                    result[f"{self.attr}_{value}"] = self.fpr[value][self.idx[value]]
            elif self.metric in ["tpr", "fnr"]:
                if self.metric == "fnr":
                    result[f"{self.attr}_{value}"] = (
                        1 - self.tpr[value][self.idx[value]]
                    )
                else:
                    result[f"{self.attr}_{value}"] = self.tpr[value][self.idx[value]]

        result = pd.DataFrame.from_dict(
            result,
            orient="index",
            columns=[f"{self.metric} at current probability threshold"],
        )

        if self.metric in ["tpr", "tnr"]:
            result["passed"] = result.iloc[:, 0].apply(lambda x: x > self.threshold)
        if self.metric in ["fpr", "fnr"]:
            result["passed"] = result.iloc[:, 0].apply(lambda x: x < self.threshold)
        result = result.round(3)

        return result

    def plot(self, save_plots: bool = True):
        """Plots ROC curve for every group in the attribute and mark the
        optimal probability threshold, the point which maximises tpr-fpr.

        Args:
          save_plots: If True, saves the plots to the class instance.
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
                label=f"{optimal_txt} = {str(round(optimal_threshold,3))}, {self.attr}_{value}",
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

    def run(self, df_test_with_output: pd.DataFrame) -> bool:
        """Runs test by calculating result and evaluating if it passes a defined
        condition.

        Args:
          df_test_with_output: Dataframe containing protected attributes with
            "prediction_probas" and "truth" column. protected attribute should
             not be encoded.
        """
        self.result = self.get_result(df_test_with_output)
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
