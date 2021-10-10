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
import numpy as np
from pandas import DataFrame
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency, norm

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class SubgroupDisparity(ModelTest):
    """
    Test if the chisq-test/maximum difference/ratio of a specified metric of any 2 groups within a specified protected attribute
    exceeds the threshold specified.

    :attr: protected attribute
    :metric: type of bias metric for the test, choose from ('fpr', 'fnr', 'pr'),
             'fpr' - false positive rate, 'fnr' - false negative rate, 'pr': positive rate
    :method: type of method for the test, choose from ('chi2', 'ratio', 'diff')
    :threshold: threshold for maximum difference/ratio or significance level of chi-sq test
    """

    attr: str
    metric: str
    method: str
    threshold: float
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Subgroup Disparity Test"
    test_desc: str = None

    def __post_init__(self):
        metrics = {
            "fpr": "false postive rate",
            "fnr": "false negative rate",
            "pr": "positive rate",
        }
        if self.metric not in metrics:
            raise AttributeError(f"metric should be one of {metrics}.")

        methods = {"diff", "ratio", "chi2"}
        if self.method not in methods:
            raise AttributeError(f"method should be one of {methods}.")

        metric_name = metrics[self.metric]
        
        if self.method == "chi2":
            default_test_desc = inspect.cleandoc(
               f"""
               Perform chi-square test of independence for {metric_name} across the subgroups.
               To pass, the p-value have to be greater than the specified significance level.
               """
            )
        else:
            default_test_desc = inspect.cleandoc(
               f"""
               Test if the maximum {self.method} of the {metric_name} of any 2 groups within
               {self.attr} attribute exceeds the threshold. To pass, this value cannot 
               exceed the threshold.
               """
            )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc
        
    def get_metric_dict(self, df: DataFrame) -> dict[str, float]:
        """Calculate metric ratio/difference and size for each subgroup of protected attribute on a given df."""   
        metric_dict = {}
        size_list = []

        for value in df[self.attr].unique():
            tmp = df[df[self.attr] == value]
            cm = confusion_matrix(tmp.truth, tmp.prediction)
            
            if self.metric == 'fnr' :
                metric_dict[value] = cm[1][0] / cm[1].sum()
                size_list.append(cm[1].sum())
            elif self.metric == 'fpr' :
                metric_dict[value] = cm[0][1] / cm[0].sum()
                size_list.append(cm[0].sum())
            elif self.metric == 'pr' :
                metric_dict[value] = (cm[1][1]+cm[0][1]) / cm.sum()
                size_list.append(cm.sum())

        return metric_dict, size_list
    
    def get_contingency_table(self, df: DataFrame) -> list:
        """Obtain the contingency table of the metric of interest for each subgroup of protected attribute on a given df,
           Compile them in a list within a list.
        """   
        table = []

        for value in df[self.attr].unique():
            tmp = df[df[self.attr] == value]
            cm = confusion_matrix(tmp.truth, tmp.prediction)
            
            if self.metric == 'fnr' :
                table.append(list(cm[1]))
            elif self.metric == 'fpr' :
                table.append(list(cm[0]))
            elif self.metric == 'pr' :
                table.append(list(cm[1]+cm[0]))

        return table

    def plot(self, alpha: float = 0.05, save_plots: bool = True):
        """
        Plot the metric of interest across the attribute subgroups, also include the confidence interval bands.
        
        :alpha: significance level for confidence interval
        :save_plots: if True, saves the plots to the class instance
        """
        
        z_value = norm.ppf(1-alpha/2)
        tmp = np.array(list(self.metric_dict.values()))
        ci = z_value*np.divide(np.multiply(tmp, 1-tmp), self.size_list)**0.5
        
        plt.figure(figsize=(12, 6))
        plt.bar(list(self.metric_dict.keys()), list(self.metric_dict.values()), yerr=ci)
        plt.axis([None, None, 0, None])
        
        title_dict = {'fpr':'False Positive Rates', 'fnr':'False Negative Rates', 'pr': 'Predicted Positive Rates'}
        title = (
            f"{title_dict[self.metric]} across {self.attr} subgroups"
        )
        plt.title(title)

        self.plots[title] = plot_to_str()
    
    def get_result_key(self) -> str:
        
        if self.method == 'chi2':
            return "p_value"
        else:
            return f"{self.attr}_{self.metric}_max_{self.method}"

    def get_result(self, df_test_with_output) -> any:
        """
        Calculate maximum ratio/diff for any 2 subgroups on a given df

        :df_test_with_output: dataframe containing protected attributes with "prediction" and "truth" column
        """
        if not {"prediction", "truth"}.issubset(df_test_with_output.columns):
            raise KeyError("df should have 'prediction' and 'truth' columns.")
        if not self.attr in set(df_test_with_output.columns):
            raise KeyError(f"Protected attribute {self.attr} column is not in given df, and ensure it is not encoded.")

        self.metric_dict, self.size_list = self.get_metric_dict(df_test_with_output)
        table = self.get_contingency_table(df_test_with_output)
        
        if self.method == 'ratio' :
            result = max(self.metric_dict.values()) / min(
                    self.metric_dict.values()
            )
        elif self.method == 'diff' :
            result = max(self.metric_dict.values()) - min(
                self.metric_dict.values()
            )
            
        elif self.method == 'chi2' :
            _,result,_,_ = chi2_contingency(table)
            
        result_key = self.get_result_key()

        return {result_key: result}

    def run(self, df_test_with_output) -> bool:
        """
        Runs test by calculating result and evaluating if it passes a defined condition.

        :df_test_with_output: dataframe containing protected attributes with "prediction" and "truth" column
                              protected attribute should not be encoded
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
