from __future__ import annotations
from dataclasses import dataclass, field
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.stats import norm

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class Permutation(ModelTest):
    """
    Check if the specified bias metric of specified attribute groups of original dataset 
    is worse than that of perturbed dataset by a specified threshold. Output a dataframe 
    showing the test result of each groups.
    i.e. Flag male gender group if
    
        FPR of male group in original data - (or division) FPR of male group in perturbed gender data > threshold

    :attr: protected attribute specified
    :metric: type of bias metric for the test, choose from ('fpr', 'fnr', 'pr'),
             'fpr' - false positive rate, 'fnr' - false negative rate, 'pr': positive rate
    :method: type of method for the test, choose from ('diff', 'ratio')
    :threshold: threshold for difference/ratio of the metric
    """

    attr: str
    metric: str
    method: str
    threshold: float
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Subgroup Permutation Test"
    test_desc: str = None

    def __post_init__(self):
        metrics = {
            "fpr": "false postive rate",
            "fnr": "false negative rate",
            "pr": "positive rate",
        }
        if self.metric not in metrics:
            raise AttributeError(f"metric should be one of {metrics}.")

        methods = {"diff", "ratio"}
        if self.method not in methods:
            raise AttributeError(f"method should be one of {methods}.")

        metric_name = metrics[self.metric]
        default_test_desc = inspect.cleandoc(
            f"""
            Test if the {self.method} of the {metric_name} of the {self.attr} subgroups of 
            the original dataset and the perturbed dataset exceeds the threshold. 
            The metric for perturbed dataset will be the {"denominator" if self.method == 'ratio' else "subtrahend"}. 
            To pass, this computed value cannot exceed the threshold.
            """
        )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    @staticmethod
    def add_predictions_to_df(df: DataFrame, model, encoder):
        """Add a column to a given df with values predicted by a given model."""
        df=df.copy()
        y_pred = model.predict(encoder.transform(df))
        df["prediction"] = y_pred
        return df
    
    def get_metric_dict(self, metric: str, df: DataFrame) -> dict[str, float]:
        """Calculate metric ratio/difference and size for each subgroup of protected attribute on a given df."""
        metric_dict = {}
        size_list = []
        
        for i in sorted(df[self.attr].unique()):
            tmp = df[df[self.attr] == i]
            cm = confusion_matrix(tmp.truth, tmp.prediction)

            if metric == "fpr":
                metric_dict[f"{self.attr}_{i}"] = cm[0][1] / cm[0].sum()
                size_list.append(cm[0].sum())
            elif metric == "fnr":
                metric_dict[f"{self.attr}_{i}"] = cm[1][0] / cm[1].sum()
                size_list.append(cm[1].sum())
            elif metric == "pr":
                metric_dict[f"{self.attr}_{i}"] = (cm[1][1]+cm[0][1]) / cm.sum()
                size_list.append(cm.sum())

        return metric_dict, size_list

    @staticmethod
    def perturb_df(attr: str, df: DataFrame):
        """Perturb the protected attribute column values of a given df."""
        df=df.copy()
        df[attr] = np.random.permutation(df[attr].values)
        return df

    def get_metric_dict_original(
        self, x_test: DataFrame, y_test: Series, model, encoder
    ):
        """Get metric dict for original dataset."""
        df_original = self.add_predictions_to_df(x_test, model, encoder)
        df_original["truth"] = y_test

        self.metric_dict_original, self.size_list_original = self.get_metric_dict(
            self.metric, df_original
        )

        return self.metric_dict_original

    def get_metric_dict_perturbed(
        self, x_test: DataFrame, y_test: Series, model, encoder
    ):
        """Get metric dict for perturbed dataset."""
        df_perturbed = self.perturb_df(self.attr, x_test)
        df_perturbed = self.add_predictions_to_df(df_perturbed, model, encoder)
        df_perturbed["truth"] = y_test

        self.metric_dict_perturbed, self.size_list_perturbed = self.get_metric_dict(
            self.metric, df_perturbed
        )

        return self.metric_dict_perturbed

    def get_result(self, x_test: DataFrame, y_test: Series, model, encoder) -> list:
        """
        Calculate test result. Compare the original vs perturbed metric
        dicts and output the attribute groups that failed the test.
        """
        if not self.attr in set(x_test.columns):
            raise KeyError(f"Protected attribute {self.attr} column is not in given df, and ensure it is not encoded.")
        
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
        Plot the metric of interest across the attribute subgroups resulting from the
        original and perturbed data respectively, also include the confidence interval bands.
        
        :alpha: significance level for confidence interval
        :save_plots: if True, saves the plots to the class instance
        """
        df_plot = self.result[[f"{self.metric} of original data",f"{self.metric} of perturbed data"]]
        
        z_value = norm.ppf(1-alpha/2)
        original_tmp = df_plot[f"{self.metric} of original data"].values
        original_ci = z_value*np.divide(np.multiply(original_tmp, 1-original_tmp), self.size_list_original)**0.5
        perturbed_tmp = df_plot[f"{self.metric} of perturbed data"].values
        perturbed_ci = z_value*np.divide(np.multiply(perturbed_tmp, 1-perturbed_tmp), self.size_list_perturbed)**0.5
                                  
        df_plot.plot.bar(yerr=[original_ci, perturbed_ci], rot=0, figsize=(12, 6))
        plt.axis([None, None, 0, None])
        
        title_dict = {'fpr':'False Positive Rates', 'fnr':'False Negative Rates', 'pr': 'Predicted Positive Rates'}
        title = (
            f"{title_dict[self.metric]} across {self.attr} subgroups"
        )
        plt.title(title)

        if save_plots:
            self.plots[title] = plot_to_str()

    def run(self, x_test: DataFrame, y_test: Series, model, encoder) -> bool:
        """
        Runs test by calculating result and evaluating if it passes a defined condition.

        :x_test: test df to be inputted into the model, protected attributes not to be encoded
        :y_test: array/list/series containing the truth of x_test
        :model: model object
        :encoder: one hot encoder object for protected attributes, must contain transform() function 
        """
        self.result = self.get_result(x_test, y_test, model, encoder)
        self.passed = False if False in list(self.result.passed) else True
        return self.passed
