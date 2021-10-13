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
import inspect
import matplotlib.pyplot as plt
import pandas as pd

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class FeatureImportance(ModelTest):
    """
    Test if the subgroups of the protected attributes are the top
    ranking important variables based on user-inputted feature 
    importance values.
    
    To pass, subgroups should not fall in the top n most important
    variables.
    
    The test also stores a dataframe showing the results of each groups.
    
    Args:
      attrs: List of protected attributes.
      threshold: Threshold for the test. To pass, subgroups should not 
         fall in the top n (threshold) most important variables.
      test_name: Name of the test, default is 'Feature Importance Test'.
      test_desc: Description of the test. If none is provided, an automatic description
         will be generated based on the rest of the arguments passed in.
    """

    attrs: list[str]
    threshold: int = 10
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Feature Importance Test"
    test_desc: str = None

    def __post_init__(self):
        default_test_desc = inspect.cleandoc(
            f"""
            Test if the subgroups of the protected attributes are the top
            ranking important variables. To pass, subgroups should not be ranked
            in the top {self.threshold} features.
            """
        )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    def plot(self, df: pd.DataFrame, show_n: int = 10, save_plots: bool = True):
        """
        Plot the top n most important features based on their importance values.
        
        Args:
          df: A dataframe with 2 columns - first column of feature names and
             second column of importance values.
          show_n: Show the top n important features on the plot.
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

    def get_result(self, df_importance) -> pd.DataFrame:
        """Output a dataframe containing the test results of the protected attributes. 
        
        Args:
          df_importance: A dataframe with 2 columns - first column with feature
             names and second column with importance values.
        """
        if df_importance.shape[1] != 2:
            raise AttributeError(
                f"There should be 2 columns in the dataframe - first column with feature names and second column with importance values"
            )

        df_importance_sorted = df_importance.sort_values(
            df_importance.columns[1], ascending=False
        ).set_index(df_importance.columns[0])
        df_importance_sorted["feature_rank"] = df_importance_sorted.iloc[:, 0].rank(
            ascending=False
        )
        df_importance_sorted = df_importance_sorted[["feature_rank"]]

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
        """Runs test by calculating result and evaluating if it passes a defined
        condition.
        
        Args:
          df_importance: A dataframe with 2 columns - first column of feature
             names and second column of importance values.
        """
        self.result = self.get_result(df_importance)
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
