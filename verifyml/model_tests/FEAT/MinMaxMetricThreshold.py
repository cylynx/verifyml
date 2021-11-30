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
from sklearn.metrics import roc_curve, mean_squared_error, mean_absolute_error
from scipy.stats import chi2

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class MinMaxMetricThreshold(ModelTest):
    """Test if all the subgroups for a given attribute meets a certain level of expected performance.

    The test also stores a dataframe showing the results of each groups. ROC curve plots
    For classification problem, plot ROC curves for every subgroup along with the points which maximises tpr-fpr.

    Args:
      attr: Column name of the protected attribute.
      metric: Type of performance metric for the test,
         For classification problem, choose from 'fpr' - false positive rate,
         'tpr' - true positive rate, 'fnr' - false negative rate, 'tnr' - true negative rate.
         For regression problem, choose from 'mse' - mean squared error, 'mae' - mean absolute error.
      threshold: Threshold for the test. To pass, fpr/fnr/mse/mae has to be lower than the threshold. tpr/tnr
         has to be greater than the threshold.
      proba_threshold: Arg for classification problem, probability threshold for the output to be classified as 1.
         By default, it is 0.5.
      test_name: Name of the test, default is 'Min Max Threshold Test'.
      test_desc: Description of the test. If none is provided, an automatic description
         will be generated based on the rest of the arguments passed in.
    """

    attr: str
    metric: Literal["fpr", "tpr", "fnr", "tnr", "mse", "mae"]
    threshold: float
    proba_threshold: float = 0.5
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Min Max Threshold Test"
    test_desc: str = None

    def __post_init__(self):
        lower_req = {"fpr", "fnr", "mse", "mae"}
        higher_req = {"tpr", "tnr"}

        if self.metric not in lower_req | higher_req:
            raise ValueError(f"metric should be one of {lower_req | higher_req}.")

        req = "lower" if self.metric in lower_req else "higher"

        default_test_desc = inspect.cleandoc(
            f"""
           Test if the {self.metric} of the subgroups within {self.attr} 
           is {req} than the threshold of {self.threshold}.
            """
        )
        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    def get_result_regression(self, df_test_with_output: pd.DataFrame) -> pd.DataFrame:
        """Test the performance metrics of the groups of protected attribute passes
        specified thresholds.

        Args:
          df_test_with_output: Dataframe containing protected attributes with
            "prediction" and "truth" column.

        Returns:
          Dataframe with the results of each group.
        """
        if not self.attr in set(df_test_with_output.columns):
            raise KeyError(
                f"Protected attribute {self.attr} column is not in given df, or is not encoded."
            )
        if not {"prediction", "truth"}.issubset(df_test_with_output.columns):
            raise KeyError(
                "Metric chosen is for regression problem, df should have 'prediction' and 'truth' columns."
            )

        result = {}
        self.dof_list = []
        for value in sorted(df_test_with_output[self.attr].unique()):
            output_sub = df_test_with_output[df_test_with_output[self.attr] == value]
            if self.metric == "mse":
                result[f"{self.attr}_{value}"] = mean_squared_error(
                    output_sub["truth"], output_sub["prediction"]
                )
            elif self.metric == "mae":
                result[f"{self.attr}_{value}"] = mean_absolute_error(
                    output_sub["truth"], output_sub["prediction"]
                )
            self.dof_list.append(len(output_sub) - 1)

        result = pd.DataFrame.from_dict(result, orient="index", columns=[self.metric])

        result["passed"] = result.iloc[:, 0].apply(lambda x: x < self.threshold)
        result = result.round(3)

        return result

    def get_result_classification(
        self, df_test_with_output: pd.DataFrame
    ) -> pd.DataFrame:
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
            raise KeyError(
                "Metric chosen is for classification problem, df should have 'prediction_probas' and 'truth' columns."
            )

        result = {}
        self.fpr = {}
        self.tpr = {}
        self.thresholds_lst = {}
        self.thresholds = {}
        self.idx = {}

        for value in sorted(df_test_with_output[self.attr].unique()):
            output_sub = df_test_with_output[df_test_with_output[self.attr] == value]
            fpr, tpr, thresholds_lst = roc_curve(
                output_sub["truth"], output_sub["prediction_probas"]
            )

            tmp = [i for i in thresholds_lst - self.proba_threshold if i > 0]
            idx = tmp.index(tmp[-1])

            self.fpr[value] = fpr
            self.tpr[value] = tpr
            self.thresholds_lst[value] = thresholds_lst
            self.thresholds[value] = self.proba_threshold
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

    def plot(self, alpha: float = 0.05, save_plots: bool = True):
        """For classification problem, plot ROC curve for every group in the attribute
        and mark the optimal probability threshold, the point which maximises tpr-fpr.
        For regression problem, plot the metric of interest across the attribute subgroups,
        and their confidence interval bands.

        Args:
          alpha: Significance level for confidence interval plot. Only applicable for regression problem.
          save_plots: If True, saves the plots to the class instance.
        """
        if self.result is None:
            raise AttributeError("Cannot plot before obtaining results.")

        if self.metric in ["mse", "mae"]:
            # Get approximate CI bounds for the metrics
            lower_list = []
            upper_list = []
            plt.figure(figsize=(12, 6))
            for i in range(len(self.dof_list)):
                metric = self.result.iloc[i, 0]
                dof = self.dof_list[i]
                if self.metric == "mse":  # mse is an unbiased estimator of sigma^2
                    tmp_lower = dof / chi2.ppf(1 - alpha / 2, df=dof)
                    tmp_higher = dof / chi2.ppf(alpha / 2, df=dof)
                elif self.metric == "mae":  # let mae be biased estimator of sigma
                    tmp_lower = np.sqrt(dof / chi2.ppf(1 - alpha / 2, df=dof))
                    tmp_higher = np.sqrt(dof / chi2.ppf(alpha / 2, df=dof))
                lower = metric * tmp_lower
                upper = metric * tmp_higher
                lower_list.append(metric - lower)
                upper_list.append(upper - metric)
            ci = [lower_list, upper_list]
            plt.bar(self.result.index, self.result.iloc[:, 0], yerr=ci)

            plt.axhline(y=self.threshold, linestyle="--", color="red")
            plt.axis([None, None, 0, None])

            title_dict = {"mse": "Mean Squared Error", "mae": "Mean Absolute Error"}
            title = f"{title_dict[self.metric]} across {self.attr} subgroups"
            plt.title(title)

        else:
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
             output and truth column. Protected attribute should not be encoded.
        """
        if self.metric in ["fpr", "fnr", "tpr", "tnr"]:
            self.result = self.get_result_classification(df_test_with_output)
        else:
            self.result = self.get_result_regression(df_test_with_output)
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
