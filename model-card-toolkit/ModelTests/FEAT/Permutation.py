from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
from sklearn.metrics import confusion_matrix
from typing import ClassVar

from ..ModelTest import ModelTest


@dataclass
class Permutation(ModelTest):
    """
    Check if the difference/ratio of specified bias metric of groups within a specified protected attribute of
    the original dataset and the perturb dataset exceeds the threshold. Output a dataframe showing the test result of each groups.
    i.e. Flag male gender group if 
    False positive rate of male group in original test data - False positive rate of male group in perturbed gender test data
    >threshold.
    Take the higher value as the numerator or the value to be subtracted from.

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

    technique: ClassVar[str] = "Permutation"

    def __post_init__(self):
        metrics = {"fpr", "fnr", "pr"}
        metric_name_dict = {"fpr":"false postive rate", "fnr":"false negative rate", "pr": "positive rate"}
        if self.metric not in metrics:
            raise AttributeError(f"metric should be one of {metrics}.")

        methods = {"diff", "ratio"}
        if self.method not in methods:
            raise AttributeError(f"method should be one of {methods}.")
        
        metric_name = metric_name_dict[self.metric]
        if self.test_name is None:
            self.test_name = "Subgroup Permutation Test"
        if self.test_desc is None:
            self.test_desc = f"Test if the {self.method} of the {metric_name} of the groups within {self.attr} attribute of the original dataset and the perturn dataset exceeds the threshold. To pass, this value cannot exceed the threshold"

    @staticmethod
    def add_predictions_to_df(df: DataFrame, model, encoder):
        """Add a column to a given df with values predicted by a given model."""
        y_pred = model.predict(df)
        df = encoder.inverse_transform(df)
        df["prediction"] = y_pred
        return df

    @staticmethod
    def get_metric_dict(attr: str, metric: str, df: DataFrame) -> dict[str, float]:
        """Calculate metric differences for a protected attribute on a given df."""
        metric_dict = {}

        for i in sorted(df[attr].unique()):
            tmp = df[df[attr] == i]
            cm = confusion_matrix(tmp.truth, tmp.prediction)

            if metric == "fpr":
                metric_dict[f"{attr}_{i}"] = cm[0][1] / cm[0].sum()
            elif metric == "fnr":
                metric_dict[f"{attr}_{i}"] = cm[1][0] / cm[1].sum()
            elif metric == "pr":
                metric_dict[f"{attr}_{i}"] = cm[1].sum() / cm.sum()

        return metric_dict

    @staticmethod
    def perturb_df(attr: str, df: DataFrame, encoder):
        """Perturb the protected attribute column values of a given df."""
        df[attr] = np.random.permutation(df[attr].values)
        df = encoder.transform(df)

        return df

    def get_metric_dict_original(
        self, x_test: DataFrame, y_test: Series, model, encoder
    ):
        """Get metric dict for original dataset."""
        df_original = self.add_predictions_to_df(x_test, model, encoder)
        df_original["truth"] = y_test

        self.metric_dict_original = self.get_metric_dict(
            self.attr, self.metric, df_original
        )

        return self.metric_dict_original

    def get_metric_dict_perturbed(
        self, x_test: DataFrame, y_test: Series, model, encoder
    ):
        """Get metric dict for perturbed dataset."""
        df_perturbed = encoder.inverse_transform(x_test)
        df_perturbed = self.perturb_df(self.attr, df_perturbed, encoder)
        df_perturbed = self.add_predictions_to_df(df_perturbed, model, encoder)
        df_perturbed["truth"] = y_test

        self.metric_dict_perturbed = self.get_metric_dict(
            self.attr, self.metric, df_perturbed
        )

        return self.metric_dict_perturbed

    def get_result(self, x_test: DataFrame, y_test: Series, model, encoder) -> list:
        """
        Calculate test result. Compare the original vs perturbed metric
        dicts and output the attribute groups that failed the test.
        """
        md_original = self.get_metric_dict_original(x_test, y_test, model, encoder)
        md_perturbed = self.get_metric_dict_perturbed(x_test, y_test, model, encoder)
        
        result = pd.DataFrame.from_dict(md_original, orient='index', columns=[f"{self.metric} of original data"])
        result[f"{self.metric} of perturbed data"] = md_perturbed.values()
        
        if self.method == "ratio":
            result['ratio'] = result[f"{self.metric} of original data"] / result[f"{self.metric} of perturbed data"]
            result['ratio'] = result.ratio.apply(lambda x: 1/x if x<1 else x)
        elif self.method == "diff":
            result['difference'] = abs(result[f"{self.metric} of original data"] - result[f"{self.metric} of perturbed data"])
        result['passed'] = result.iloc[:,-1] < self.threshold
        result = result.round(3)
        return result

    def run(self, x_test: DataFrame, y_test: Series, model, encoder) -> bool:
        """
        Runs test by calculating result and evaluating if it passes a defined condition.

        :x_test: dataframe containing features to be inputted into the model predictions
        :y_test: array/list/series containing the truth of x_test
        :model: model object
        :encoder: one hot encoder object, to allow for permutation of the protected attribute
        """
        self.result = self.get_result(x_test, y_test, model, encoder)
        self.passed = False if False in list(self.result.passed) else True
        return self.passed
