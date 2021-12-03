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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.base import is_classifier
from scipy.stats import norm, chi2

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class Perturbation(ModelTest):
    """
    Test if the specified metric of specified attribute subgroups of original
    dataset is worse than that of perturbed dataset by a specified threshold.

    To pass, if ratio is used, the ratio (with metric of perturbed data as denominator)
    of the respective subgroups metrics of the datasets should not exceed the threshold.

    If diff is used, the difference (with metric of perturbed data as subtrahend)
    of the respective subgroups metric of the datasets should not exceed the threshold.

    The test also stores a dataframe showing the results of each groups.

    Args:
      attr: Column name of the protected attribute.
      metric: Type of performance metric for the test,
         For classification problem, choose from 'fpr' - false positive rate,
         'fnr' - false negative rate, 'pr' - positive rate.
         For regression problem, choose from 'mse' - mean squared error, 'mae' - mean absolute error.
      method: Type of method for the test, choose from 'ratio' or 'diff'.
      threshold: Threshold for the test. To pass, ratio/difference of chosen metric has
         to be lower than the threshold.
      proba_threshold: Arg for classification problem, probability threshold for the output to be classified as 1.
         By default, it is 0.5.
      test_name: Name of the test, default is 'Subgroup Perturbation Test'.
      test_desc: Description of the test. If none is provided, an automatic description
         will be generated based on the rest of the arguments passed in.
    """

    attr: str
    metric: Literal["fpr", "fnr", "pr", "mse", "mae"]
    method: Literal["ratio", "diff"]
    threshold: float
    proba_threshold: float = 0.5
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Subgroup Perturbation Test"
    test_desc: str = None

    def __post_init__(self):
        metrics = {
            "fpr": "false postive rate",
            "fnr": "false negative rate",
            "pr": "positive rate",
            "mse": "mean squared error",
            "mae": "mean absolute error",
        }
        if self.metric not in metrics:
            raise AttributeError(f"metric should be one of {metrics}.")

        methods = {"diff", "ratio"}
        if self.method not in methods:
            raise AttributeError(f"method should be one of {methods}.")

        metric_name = metrics[self.metric]
        default_test_desc = inspect.cleandoc(
            f"""
            Test if the {self.method} of the {metric_name} of the {self.attr}
            subgroups of the original dataset and the perturbed dataset exceeds
            the threshold. The metric for perturbed dataset will be the
            {"denominator" if self.method == 'ratio' else "subtrahend"}. To
            pass, this computed value cannot exceed {self.threshold}.
            """
        )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    def add_predictions_to_df(self, df: pd.DataFrame, model, encoder) -> pd.DataFrame:
        """
        Predict a set of dataset using the given model, and output the predictions
        together with the df. Before predicting, encode the categorical features
        in the dataset with the encoder object.

        Args:
          df: Dataset to be predicted by the model, protected attributes not
             to be encoded
          model: Model class object, preferably Sklearn class.
          encoder: One hot encoder class object for protected, preferably Sklearn class,
             must contain transform() function.
        """
        df = df.copy()
        if not is_classifier(model):
            y_pred = model.predict(encoder.transform(df))
        else:
            y_pred = (
                model.predict_proba(encoder.transform(df))[::, 1] > self.proba_threshold
            )
        df["prediction"] = y_pred
        return df

    def get_metric_dict(self, df: pd.DataFrame) -> Tuple[dict, list]:
        """
        Output a dictionary containing the metrics and a list of the
        metric's sample size for each subgroup of protected attribute,
        from a dataframe containing 'truth' and 'prediction' columns.

        Args:
          df: Dataframe containing 'truth' and 'prediction' columns.
        """
        metric_dict = {}
        size_list = []

        for i in sorted(df[self.attr].unique()):
            tmp = df[df[self.attr] == i]

            if self.metric in ["fpr", "fnr", "pr"]:
                cm = confusion_matrix(tmp.truth, tmp.prediction)
            if self.metric == "fpr":
                metric_dict[f"{self.attr}_{i}"] = cm[0][1] / cm[0].sum()
                size_list.append(cm[0].sum())
            elif self.metric == "fnr":
                metric_dict[f"{self.attr}_{i}"] = cm[1][0] / cm[1].sum()
                size_list.append(cm[1].sum())
            elif self.metric == "pr":
                metric_dict[f"{self.attr}_{i}"] = (cm[1][1] + cm[0][1]) / cm.sum()
                size_list.append(cm.sum())
            elif self.metric == "mse":
                metric_dict[f"{self.attr}_{i}"] = mean_squared_error(
                    tmp["truth"], tmp["prediction"]
                )
                size_list.append(len(tmp) - 1)
            elif self.metric == "mae":
                metric_dict[f"{self.attr}_{i}"] = mean_absolute_error(
                    tmp["truth"], tmp["prediction"]
                )
                size_list.append(len(tmp) - 1)

        return metric_dict, size_list

    @staticmethod
    def perturb_df(attr: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perturb (by shuffling) the protected attribute column values of a
        given df and and output a new dataframe.

        Args:
          attr: Column name of the protected attribute to be perturbed.
          df: Dataframe containing the protected attribute.
        """
        df = df.copy()
        df[attr] = np.random.permutation(df[attr].values)
        return df

    def get_metric_dict_original(
        self, x_test: pd.DataFrame, y_test: pd.Series, model, encoder
    ) -> dict[str]:
        """Get metric dict for original dataset.

        Args:
          x_test: Test dataset to be predicted by the model, protected attributes not
             to be encoded
          y_test: Series/array/list containing the truth outcome of x_test
          model: Model class object, preferably Sklearn class.
          encoder: One hot encoder class object for protected, preferably Sklearn class,
             must contain transform() function.
        """
        df_original = self.add_predictions_to_df(x_test, model, encoder)
        df_original["truth"] = y_test

        self.metric_dict_original, self.size_list_original = self.get_metric_dict(
            df_original
        )

        return self.metric_dict_original

    def get_metric_dict_perturbed(
        self, x_test: pd.DataFrame, y_test: pd.Series, model, encoder
    ) -> dict[str]:
        """Get metric dict for perturbed dataset.

        Args:
          x_test: Test dataset to be perturbed and predicted by the model,
             protected attributes not to be encoded
          y_test: Series/array/list containing the truth outcome of x_test
          model: Model class object, preferably Sklearn class.
          encoder: One hot encoder class object for protected, preferably Sklearn class,
             must contain transform() function.
        """
        df_perturbed = self.perturb_df(self.attr, x_test)
        df_perturbed = self.add_predictions_to_df(df_perturbed, model, encoder)
        df_perturbed["truth"] = y_test

        self.metric_dict_perturbed, self.size_list_perturbed = self.get_metric_dict(
            df_perturbed
        )

        return self.metric_dict_perturbed

    def get_result(
        self, x_test: pd.DataFrame, y_test: pd.Series, model, encoder
    ) -> pd.DataFrame:
        """
        Output a dataframe showing the test result of each groups.
        For an example in 'gender' attribute, male subgroup fail the test if
            FPR of male group in original data - (or division) FPR of male group in
            perturbed gender data > threshold.

        Args:
          x_test: Test df to be inputted into the model, protected attributes not
             to be encoded
          y_test: Series/array/list containing the truth of x_test
          model: Model class object, preferably Sklearn class
          encoder: One hot encoder class object, preferably Sklearn class
             attributes, must contain transform() function
        """
        if not self.attr in set(x_test.columns):
            raise KeyError(
                f"Protected attribute {self.attr} column is not in given df, and ensure it is not encoded."
            )
        if not is_classifier(model) and self.metric not in ["mse", "mae"]:
            raise ValueError(
                f"Classification metrics is not applicable with regression problem. Try metric = 'mse' "
            )

        md_original = self.get_metric_dict_original(x_test, y_test, model, encoder)
        md_perturbed = self.get_metric_dict_perturbed(x_test, y_test, model, encoder)

        result = pd.DataFrame.from_dict(
            md_original, orient="index", columns=[f"{self.metric} of original data"]
        )
        result[f"{self.metric} of perturbed data"] = md_perturbed.values()

        if self.method == "ratio":
            result["ratio"] = (
                result[f"{self.metric} of original data"]
                / result[f"{self.metric} of perturbed data"]
            )
        elif self.method == "diff":
            result["difference"] = (
                result[f"{self.metric} of original data"]
                - result[f"{self.metric} of perturbed data"]
            )
        result["passed"] = result.iloc[:, -1] <= self.threshold
        result = result.round(3)
        return result

    def plot(self, alpha: float = 0.05, save_plots: bool = True):
        """
        Plot the metrics of the attribute subgroups resulting from the
        original and perturbed data respectively, also include the
        confidence interval bands.

        Args:
          alpha: Significance level for confidence interval.
          save_plots: If True, saves the plots to the class instance.
        """
        df_plot = self.result[
            [f"{self.metric} of original data", f"{self.metric} of perturbed data"]
        ]

        if self.metric in ["mse", "mae"]:
            # Get approximate CI bounds for the metrics
            lower_list = []
            upper_list = []
            for i in range(len(self.size_list_original)):
                dof = self.size_list_original[i]
                metric = df_plot[f"{self.metric} of original data"].values[i]
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
            original_ci = [lower_list, upper_list]
        else:
            z_value = norm.ppf(1 - alpha / 2)
            original_tmp = df_plot[f"{self.metric} of original data"].values
            original_ci = (
                z_value
                * np.divide(
                    np.multiply(original_tmp, 1 - original_tmp), self.size_list_original
                )
                ** 0.5
            )

        if self.metric in ["mse", "mae"]:
            # Get approximate CI bounds for the metrics
            lower_list = []
            upper_list = []
            for i in range(len(self.size_list_perturbed)):
                dof = self.size_list_perturbed[i]
                metric = df_plot[f"{self.metric} of original data"].values[i]
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
            perturbed_ci = [lower_list, upper_list]
        else:
            perturbed_tmp = df_plot[f"{self.metric} of perturbed data"].values
            perturbed_ci = (
                z_value
                * np.divide(
                    np.multiply(perturbed_tmp, 1 - perturbed_tmp),
                    self.size_list_perturbed,
                )
                ** 0.5
            )

        df_plot.plot.bar(yerr=[original_ci, perturbed_ci], rot=0, figsize=(12, 6))
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

    def run(self, x_test: pd.DataFrame, y_test: pd.Series, model, encoder) -> bool:
        """Runs test by calculating result and evaluating if it passes a defined
        condition.

        Args:
          x_test: Test df to be inputted into the model, protected attributes not
             to be encoded
          y_test: Series/array/list containing the truth of x_test
          model: Model class object, preferably Sklearn class
          encoder: One hot encoder class object, preferably Sklearn class
             attributes, must contain transform() function
        """
        self.result = self.get_result(x_test, y_test, model, encoder)
        self.passed = False if False in list(self.result.passed) else True
        return self.passed
