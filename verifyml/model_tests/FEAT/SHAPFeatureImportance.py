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
import pandas as pd
import shap

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class SHAPFeatureImportance(ModelTest):
    """
    Test if the subgroups of the protected attributes are the top
    ranking important variables under shapely feature importance value. 
    
    To pass, subgroups should not fall in the top n most important
    variables.
    
    The test also stores a dataframe showing the results of each groups.
    
    Args:
      attrs: List of protected attributes.
      threshold: Threshold for the test. To pass, subgroups should not 
         fall in the top n (threshold) most important variables.
      test_name: Name of the test, default is 'Shapely Feature Importance Test'.
      test_desc: Description of the test. If none is provided, an automatic description
         will be generated based on the rest of the arguments passed in.
    """

    attrs: list[str]
    threshold: int = 10
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Shapely Feature Importance Test"
    test_desc: str = None

    def __post_init__(self):
        default_test_desc = inspect.cleandoc(
            f"""
           Test if the subgroups of the protected attributes are the top ranking
           influential variables under shapely feature importance value. To
           pass, subgroups should not be ranked in the top {self.threshold}
           features.
        """
        )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    def get_shap_values(self, model, model_type, x_train_encoded, x_test_encoded) -> list:
        '''
        Get SHAP values for a set of test samples.
        
        Args:
          model: Trained model object.
          model_type: type of model algorithm, choose from 'trees' or 'others' 
          x_train_encoded: Training data features, categorical features have to be encoded.
          x_test_encoded: Test data to be used for shapely explanations, categorical features have
             to be encoded.
        '''
        if model_type == "trees":
            explainer = shap.TreeExplainer(
                model=model,
                model_output="margin",
                feature_perturbation="tree_path_dependent",
            )
        elif model_type == "others":
            explainer = shap.PermutationExplainer(
                model=model.predict_proba, data=x_train_encoded
            )
        else:
            raise ValueError("model_type should be 'trees' or 'others'")

        self.shap_values = explainer.shap_values(x_test_encoded)

        return self.shap_values

    def shap_summary_plot(self, x_test_encoded, save_plots: bool = True):
        """Make a shap summary plot.
        
        Args:
          x_test_encoded: Data to be used for shapely explanations, categorical
             features have to be encoded 
          save_plots: if True, saves the plots to the class instance
        """
        shap.summary_plot(
            shap_values=self.shap_values[1],
            features=x_test_encoded,
            max_display=20,
            plot_type="dot",
            show=False,
        )

        if save_plots:
            self.plots["SHAP Summary Plot"] = plot_to_str()
        else:
            plot_to_str()

    def get_result(
        self,
        model,
        model_type: str,
        x_train_encoded: pd.DataFrame,
        x_test_encoded: pd.DataFrame,
    ) -> pd.DataFrame:
        """Output a dataframe containing the test results of the protected attributes. 
 
        Args:
          model: Trained model object.
          model_type: type of model algorithm, choose from 'trees' or 'others' 
          x_train_encoded: Training data features, categorical features have to be encoded.
          x_test_encoded: Test data to be used for shapely explanations, categorical features have
             to be encoded.
        """
        if ("object" in list(x_train_encoded.dtypes)) or (
            "object" in list(x_test_encoded.dtypes)
        ):
            raise AttributeError(f"Categorical features have to be encoded.")

        shap_values = self.get_shap_values(
            model, model_type, x_train_encoded, x_test_encoded
        )

        # Take the mean of the absolute shapely values to get the aggregated shapely importance for each features
        agg_shap_df = pd.DataFrame(
            pd.DataFrame(shap_values[1], columns=x_test_encoded.columns).abs().mean()
        ).sort_values(0, ascending=False)
        agg_shap_df["feature_rank"] = agg_shap_df[0].rank(ascending=False)
        agg_shap_df.drop(0, axis=1, inplace=True)
        attrs_string = "|".join([f"{x}_" for x in self.attrs])
        result = agg_shap_df[
            agg_shap_df.index.to_series().str.contains(attrs_string)
        ].copy()
        result["passed"] = result.feature_rank.apply(
            lambda x: True if x > self.threshold else False
        )

        return result

    def shap_dependence_plot(
        self, x_test_encoded, show_all: bool = True, save_plots: bool = True
    ):
        """
        Create a SHAP partial dependence plot to show the effect of the
        individual subgroups on shapely value.
        
        Args:
          x_test_encoded: Test data to be used for shapely explanations, categorical
             features have to be encoded.
          show_all: If false, only show subgroups that failed the test.
        """
        if self.result is None:
            raise AttributeError("Cannot create dependence plot before running test.")
        if show_all:
            attrs_to_show = self.result.index
        else:
            attrs_to_show = self.result[self.result.passed == False].index

        for r in attrs_to_show:
            shap.dependence_plot(
                r,
                shap_values=self.shap_values[1],
                features=x_test_encoded,
                interaction_index=None,
                show=False,
            )

            if save_plots:
                self.plots[f"SHAP Dependence Plot: {r}"] = plot_to_str()
            else:
                plot_to_str()

    def run(
        self,
        model,
        model_type: Literal["trees", "others"],
        x_train_encoded: pd.DataFrame,
        x_test_encoded: pd.DataFrame,
    ) -> bool:
        """Runs test by calculating result and evaluating if it passes a defined
        condition.
        
        Args:
          model: Trained model object.
          model_type: type of model algorithm, choose from 'trees' or 'others' 
          x_train_encoded: Training data features, categorical features have to be encoded.
          x_test_encoded: Test data to be used for shapely explanations, categorical features have
             to be encoded.
        """
        self.result = self.get_result(
            model, model_type, x_train_encoded, x_test_encoded
        )
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
