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
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, norm

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class DataShift(ModelTest):
    """
    Test if there is any shift (based on specified threshold and method) in the distribution of the protected feature,
    which may impose new unfairness and require a retraining of the model, output the dataframe detailing whether the attribute
    passed. Take the higher value as the numerator or the value to be subtracted from.

    :protected_attr: list of protected attributes
    :method: type of method for the test, choose from ('chi2', 'ratio', 'diff')
    :threshold: probability distribution threshold or significance level of chi-sq test
    """

    protected_attr: list[str]
    method: str = "chi2"
    threshold: float = 1.25
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Data Shift Test"
    test_desc: str = None

    def __post_init__(self):
        
        if self.method == "chi2":
            pass_desc = f"""
                        To pass, the p-value have to be greater than the significance 
                        level of chi-square test of independence between the datasets 
                        and the attribute distribution, as specified by the threshold argument.
                        """
        else:
            pass_desc = f"""
                        To pass, the {self.method} of distribution of the subgroup in the 
                        training data and evaluation data should not exceed the threshold.
                        """
            
        default_test_desc = inspect.cleandoc(
            f"""
            Test if there is any shift in the distribution of the attribute subgroups across the different datasets. {pass_desc}
            """
        )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    @staticmethod
    def get_df_distribution_by_pa(df: DataFrame, col: str, freq: bool = False):
        """
        Get the probability distribution of a specified column's values in a given df.
        
        :freq: output the contingency table if true else output the probabilites
        """
        if freq:
            df_dist = df.groupby(col)[col].apply(lambda x: x.count())
        else:
            df_dist = df.groupby(col)[col].apply(lambda x: x.count() / len(df))

        return df_dist

    def get_result(self, x_train: DataFrame, x_test: DataFrame) -> any:
        """
        Calculate test result.

        :x_train: training data features, protected features should not be encoded 
        :x_test: data to be evaluated on, protected features should not be encoded 
        """
        if (not set(self.protected_attr).issubset(x_train.columns)) or (not set(self.protected_attr).issubset(x_test.columns)):
            raise KeyError(f"Protected attribute columns {set(self.protected_attr)} are not in given df, and ensure they are not encoded.")
            
        result = pd.DataFrame()
        for pa in self.protected_attr:
            train_dist = pd.DataFrame(self.get_df_distribution_by_pa(x_train, pa, False))
            eval_dist = pd.DataFrame(self.get_df_distribution_by_pa(x_test, pa, False))
  
            df_dist=pd.concat([train_dist,eval_dist],axis=1)
            df_dist.columns = ['training_distribution', 'eval_distribution']
            df_dist.index = df_dist.index.to_series().apply(lambda x: f"{pa}_{x}")
            
            if self.method == "chi2":
                train_freq = pd.DataFrame(self.get_df_distribution_by_pa(x_train, pa, True))
                eval_freq = pd.DataFrame(self.get_df_distribution_by_pa(x_test, pa, True))
                df_freq=pd.concat([train_freq,eval_freq],axis=1)
                _,p,_,_=chi2_contingency(df_freq.values)
                df_dist["p-value"] = p
                    
            elif self.method == "ratio":
                df_dist["ratio"] = (
                    df_dist["training_distribution"]
                    / df_dist["eval_distribution"]
                )
                df_dist["ratio"] = df_dist.ratio.apply(
                    lambda x: 1 / x if x < 1 else x
                )
            elif self.method == "diff":
                df_dist["difference"] = abs(
                    df_dist["training_distribution"]
                    - df_dist["eval_distribution"]
                )
            result = result.append(df_dist)
        
        if self.method == "chi2":
            result["passed"] = result.iloc[:, -1] > self.threshold
        else:
            result["passed"] = result.iloc[:, -1] <= self.threshold
        result = result.round(3)

        return result

    def plot(self, alpha: float = 0.05, save_plots: bool = True):
        """
        Plot the the probability distribution of subgroups of protected attribute for
        training and evaluation data respectively, also include the confidence interval bands.
        
        :alpha: significance level for confidence interval
        :save_plots: if True, saves the plots to the class instance
        """
        fig, axs = plt.subplots(
            1,
            len(self.protected_attr),
            figsize=(18, 6),
        )
        num = 0
        for pa in self.protected_attr:
            df_plot = self.result[['training_distribution','eval_distribution']]
            df_plot = df_plot[df_plot.index.to_series().str.contains(f"{pa}_")]
            
            z_value = norm.ppf(1-alpha/2)
            train_ci = list(df_plot['training_distribution'].apply(lambda x: z_value*(x*(1-x)/self.df_size[0])**0.5))
            eval_ci = list(df_plot['eval_distribution'].apply(lambda x: z_value*(x*(1-x)/self.df_size[1])**0.5))
            
            df_plot.plot.bar(yerr=[train_ci, eval_ci], rot=0, ax=axs[num], title=pa)
            num+=1

        title = (
            "Probability Distribution of protected attributes"
        )
        fig.suptitle(title)

        if save_plots:
            self.plots[title] = plot_to_str()

    def run(self, x_train: DataFrame, x_test: DataFrame) -> bool:
        """
        Runs test by calculating result / retrieving cached property and evaluating if
        it passes a defined condition.

        :x_train: training data features, protected features should not be encoded
        :x_test: data to be evaluated on, protected features should not be encoded
        """
        self.result = self.get_result(x_train, x_test)
        self.df_size = [len(x_train), len(x_test)]
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
