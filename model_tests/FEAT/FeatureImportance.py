from __future__ import annotations
from dataclasses import dataclass, field
import inspect
import matplotlib.pyplot as plt
from pandas import DataFrame

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class FeatureImportance(ModelTest):
    """
    Ouput a dataframe consisting of protected attributes and its respective ranking based on user-inputted feature importance values.
    To pass, subgroups of protected attributes should not fall in the top n most important variables.

    :attrs: protected attributes
    :threshold: the top n features to be specified
    """

    attrs: list[str]
    threshold: int = 10
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Feature Importance Test"
    test_desc: str = None

    def __post_init__(self):
        default_test_desc = inspect.cleandoc(
            f"""
           Test if the subgroups of the protected attributes are the top ranking important
           variables. To pass, subgroups should not be ranked in the top {self.threshold} 
           features.
        """
        )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    def plot(self, df: DataFrame, show_n: int = 10, save_plots: bool = True):
        """
        :df: A dataframe with 2 columns - first column of feature names and second column of importance values
        :show_n: Show the top n important features on the plot
        """
        title = "Feature Importance Plot"
        df_sorted = df.sort_values(df.columns[1], ascending=False)
        # Plot top n important features
        plt.figure(figsize=(15, 8))
        plt.barh(df_sorted.iloc[:show_n, 0], df_sorted.iloc[:show_n, 1])
        plt.gca().invert_yaxis()
        plt.title(title)
        plt.xlabel("Relative Importance Value")
        plt.tight_layout()

        if save_plots:
            self.plots[title] = plot_to_str()

    def get_result(self, df_importance) -> any:
        """
        Output the protected attributes that are listed in the top specified number of the features,
        using feature importance values inputted by the user.

        :df_importance: A dataframe with 2 columns - first column of feature names and second column of importance values
        """
        df_importance_sorted = df_importance.sort_values(
            df_importance.columns[1], ascending=False
        ).set_index(df_importance.columns[0])
        df_importance_sorted["feature_rank"] = df_importance_sorted.iloc[:, 0].rank(
            ascending=False
        )
        df_importance_sorted = df_importance_sorted[['feature_rank']]

        attrs_string = "|".join([f"{x}_" for x in self.attrs])
        result = df_importance_sorted[
            df_importance_sorted.index.to_series().str.contains(attrs_string)
        ].copy()
        result["passed"] = result.feature_rank.apply(
            lambda x: True if x > self.threshold else False
        )
        result.index.name = None
        return result

    def run(self, df_importance) -> bool:
        """
        Runs test by calculating result and evaluating if it passes a defined condition.

        :df_importance: A dataframe with 2 columns - first column of feature names and second column of importance values
        """
        self.result = self.get_result(df_importance)
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
