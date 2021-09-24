from __future__ import annotations
from dataclasses import dataclass, field
from pandas import DataFrame
import shap
from typing import ClassVar

from .FEATTest import FEATTest
from .utils import plot_to_str


@dataclass
class SHAPFeatureImportance(FEATTest):
    """
    Ouput the protected attributes that are listed in the top specified % of the features influencing the predictions
    ,using aggregated shapely values.

    :attrs: list of protected attributes
    :top_n: the top n features to be specified
    """

    attrs: list[str]
    top_n: int
    plots: dict[str, str] = field(repr=False, default_factory=lambda: {})

    technique: ClassVar[str] = "SHAP Feature Importance"

    def get_shap_values(self, model, model_type, x_train, x_test):
        if model_type == "trees":
            explainer = shap.TreeExplainer(model=model, model_output="margin")
        elif model_type == "others":
            explainer = shap.PermutationExplainer(
                model=model.predict_proba, data=x_train
            )
        else:
            raise ValueError("model_type should be 'trees' or 'others'")

        self.shap_values = explainer.shap_values(x_test)

        return self.shap_values

    def shap_summary_plot(self, x_test):
        """
        Make a shap summary plot.

        :x_test: data to be used for shapely explanations, preferably eval set, categorical features have to be already encoded
        """
        shap.summary_plot(
            shap_values=self.shap_values[1],
            features=x_test,
            max_display=20,
            plot_type="dot",
        )

        self.plots["SHAP Summary Plot"] = plot_to_str()

    def get_result(
        self, model, model_type: str, x_train: DataFrame, x_test: DataFrame
    ) -> list:
        """
        Output the protected attributes that are listed in the top specified % of the features influencing the predictions
        , using aggregated shapely values.
        """
        result = []
        shap_values = self.get_shap_values(model, model_type, x_train, x_test)

        # Take the mean of the absolute of the shapely values to get the aggretated importance for each features
        agg_shap_df = DataFrame(
            DataFrame(shap_values[1], columns=x_test.columns).abs().mean()
        ).sort_values(0, ascending=False)
        top_feat = list(agg_shap_df.iloc[: self.top_n].index)

        for attr in self.attrs:
            result += [feat for feat in top_feat if f"{attr}_" in feat]

        return result

    def shap_dependence_plot(self, x_test):
        """
        Create a SHAP dependence plot to show the significant effect of the flagged
        protected attributes across the whole dataset.
        """
        if self.result is None:
            raise AttributeError("Cannot create dependence plot before running test.")

        for r in self.result:
            shap.dependence_plot(
                r,
                shap_values=self.shap_values[1],
                features=x_test,
                interaction_index=None,
            )
            self.plots[f"SHAP Dependence Plot: {r}"] = plot_to_str()

    def run(
        self, model, model_type: str, x_train: DataFrame, x_test: DataFrame
    ) -> bool:
        """
        Runs test by calculating result and evaluating if it passes a defined condition.

        :model: trained model object
        :model_type: type of algorithim, choose from ['trees','others']
        :x_train: training data features, categorical features have to be already encoded
        :x_test: data to be used for shapely explanations, preferably eval set, categorical features have to be already encoded
        """
        self.result = self.get_result(model, model_type, x_train, x_test)
        self.passed = False if self.result else True

        return self.passed
