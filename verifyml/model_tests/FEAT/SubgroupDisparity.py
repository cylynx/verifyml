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
from typing import Literal, Tuple
import inspect
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from scipy.stats import chi2_contingency, norm, chi2

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class SubgroupDisparity(ModelTest):
    """
    Test if the maximum difference / ratio of a specified metric for any 2
    groups within a specified protected attribute exceeds the given threshold.

    If chi2 is used, the p-value calculated from a chi-square test of
    independence should be greater than the level of significance as specified
    by the threshold.

    Args:
      attr: Column name of the protected attribute.
      metric: Type of performance metric for the test,
        For classification problem, choose from 'fpr' - false positive rate,
        'fnr' - false negative rate, 'pr' - positive rate.
        For regression problem, choose from 'mse' - mean squared error, 'mae' - mean absolute error.
      method: Type of method for the test, choose from 'chi2', 'ratio' or 'diff'.
      threshold: Threshold for maximum difference / ratio, or the significance level of chi-sq test.
      test_name: Name of the test, default is 'Subgroup Disparity Test'.
      test_desc: Description of the test. If none is provided, an automatic description
         will be generated based on the rest of the arguments passed in.
    """

    attr: str
    metric: Literal["fpr", "fnr", "pr", "mse", "mae"]
    method: Literal["chi2", "ratio", "diff"]
    threshold: float
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Subgroup Disparity Test"
    test_desc: str = None

    def __post_init__(self):
        metrics = {
            "fpr": "false postive rate",
            "fnr": "false negative rate",
            "pr": "positive rate",
            "mse": "mean squared error",
            "mae": "Mean Absolute Error",
        }
        if self.metric not in metrics:
            raise ValueError(f"metric should be one of {metrics}.")

        methods = {"diff", "ratio", "chi2"}
        if self.method not in methods:
            raise ValueError(f"method should be one of {methods}.")

        metric_name = metrics[self.metric]

        if self.method == "chi2":
            default_test_desc = inspect.cleandoc(
                f"""
               To pass, the p-value calculated from a chi-square test of
               independence for {metric_name} across the subgroups should be
               greater than {self.threshold*100}% significance level.
               """
            )
        else:
            default_test_desc = inspect.cleandoc(
                f"""
               Test if the maximum {self.method} of the {metric_name} of any 2
               groups within {self.attr} attribute exceeds {self.threshold}. To
               pass, this value cannot exceed the threshold.
               """
            )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    def get_metric_dict(self, df: pd.DataFrame) -> Tuple[dict, list]:
        """Calculate metric ratio / difference and size for each subgroup of the
        protected attribute on a given df.

        Args:
          df: Dataframe.

        Returns:
          A dictionary of each subgroup and the calculated ratio or difference.
        """
        metric_dict = {}
        size_list = []

        for value in sorted(df[self.attr].unique()):
            tmp = df[df[self.attr] == value]

            if self.metric in ["fpr", "fnr", "pr"]:
                cm = confusion_matrix(tmp.truth, tmp.prediction)
            if self.metric == "fnr":
                metric_dict[value] = cm[1][0] / cm[1].sum()
                size_list.append(cm[1].sum())
            elif self.metric == "fpr":
                metric_dict[value] = cm[0][1] / cm[0].sum()
                size_list.append(cm[0].sum())
            elif self.metric == "pr":
                metric_dict[value] = (cm[1][1] + cm[0][1]) / cm.sum()
                size_list.append(cm.sum())
            elif self.metric == "mse":
                metric_dict[value] = mean_squared_error(tmp["truth"], tmp["prediction"])
                size_list.append(len(tmp) - 1)
            elif self.metric == "mae":
                metric_dict[value] = mean_absolute_error(
                    tmp["truth"], tmp["prediction"]
                )
                size_list.append(len(tmp) - 1)

        return metric_dict, size_list

    def get_contingency_table(self, df: pd.DataFrame) -> list:
        """Obtain the contingency table of the metric of interest for each subgroup
        of a protected attribute on a given df.

        Args:
          df: Dataframe.

        Returns:
          List of metric value.
        """
        table = []

        for value in df[self.attr].unique():
            tmp = df[df[self.attr] == value]
            cm = confusion_matrix(tmp.truth, tmp.prediction)

            if self.metric == "fnr":
                table.append(list(cm[1]))
            elif self.metric == "fpr":
                table.append(list(cm[0]))
            elif self.metric == "pr":
                table.append(list(cm[1] + cm[0]))

        return table

    def plot(self, alpha: float = 0.05, save_plots: bool = True):
        """Plot the metric of interest across the attribute subgroups, and their
        confidence interval bands.

        Args:
          alpha: Significance level for confidence interval.
          save_plots: If True, saves the plots to the class instance.
        """
        if self.metric in ["mse", "mae"]:
            # Get approximate CI bounds for the metrics
            lower_list = []
            upper_list = []
            for i in range(len(self.size_list)):
                dof = self.size_list[i]
                metric = list(self.metric_dict.values())[i]
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
        else:
            z_value = norm.ppf(1 - alpha / 2)
            tmp = np.array(list(self.metric_dict.values()))
            ci = z_value * np.divide(np.multiply(tmp, 1 - tmp), self.size_list) ** 0.5

        plt.figure(figsize=(12, 6))
        plt.bar(list(self.metric_dict.keys()), list(self.metric_dict.values()), yerr=ci)
        plt.axis([None, None, 0, None])

        title_dict = {
            "fpr": "False Positive Rates",
            "fnr": "False Negative Rates",
            "pr": "Predicted Positive Rates",
            "mse": "Mean Squared Error",
            "mae": "Mean Absolute Error",
        }
        title = f"{title_dict[self.metric]} across {self.attr} subgroups"
        plt.title(title)

        if save_plots:
            self.plots[title] = plot_to_str()

    def get_result_key(self) -> str:

        if self.method == "chi2":
            return "p_value"
        else:
            return f"{self.attr}_{self.metric}_max_{self.method}"

    def get_result(self, df_test_with_output: pd.DataFrame) -> Dict[str, float]:
        """Calculate maximum ratio / diff or chi-sq test for any 2 subgroups' metrics on a
        given df.

        Args:
          df_test_with_output: Dataframe containing protected attributes with "prediction" and "truth" column.
        """
        if not {"prediction", "truth"}.issubset(df_test_with_output.columns):
            raise KeyError("df should have 'prediction' and 'truth' columns.")
        if not self.attr in set(df_test_with_output.columns):
            raise KeyError(
                f"Protected attribute {self.attr} column is not in given df, or is not encoded."
            )
        if df_test_with_output.truth.nunique() != 2 and self.metric in [
            "fpr",
            "fnr",
            "pr",
        ]:
            raise ValueError(
                f"Classification metrics is not applicable with regression problem. Try metric = 'mse' "
            )

        self.metric_dict, self.size_list = self.get_metric_dict(df_test_with_output)
        if self.method == "ratio":
            result = max(self.metric_dict.values()) / min(self.metric_dict.values())
        elif self.method == "diff":
            result = max(self.metric_dict.values()) - min(self.metric_dict.values())
        elif self.method == "chi2":
            if self.metric in ["mse", "mae"]:
                raise ValueError(
                    f"Chi-square test is not applicable for regression metrics, try method = 'ratio'. "
                )
            table = self.get_contingency_table(df_test_with_output)
            _, result, _, _ = chi2_contingency(table)

        result_key = self.get_result_key()

        return {result_key: result}

    def run(self, df_test_with_output: pd.DataFrame) -> bool:
        """Runs test by calculating result and evaluating if it passes a defined
        condition.

        Args:
          df_test_with_output: Dataframe containing protected attributes with
            "prediction_probas" and "truth" column. protected attribute should
             not be encoded.
        """
        self.result = self.get_result(df_test_with_output)

        result_value = self.result[self.get_result_key()]

        if self.method == "chi2":
            self.passed = True if result_value > self.threshold else False
        else:
            self.passed = True if result_value <= self.threshold else False

        # Convert result object into DataFrame
        self.result = pd.DataFrame(self.result.values(), columns=self.result.keys())
        self.result = self.result.round(3)

        return self.passed
