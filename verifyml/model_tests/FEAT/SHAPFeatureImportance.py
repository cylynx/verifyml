from __future__ import annotations
from dataclasses import dataclass, field
import inspect
from pandas import DataFrame
import shap

from ..ModelTest import ModelTest
from ..utils import plot_to_str


@dataclass
class SHAPFeatureImportance(ModelTest):
    """
    Ouput a dataframe consisting of protected attributes and its respective ranking based on aggregated shapely value.
    To pass, subgroups of protected attributes should not fall in the top n most important variables.

    :attrs: list of protected attributes
    :threshold: the top n features to be specified
    """

    attrs: list[str]
    threshold: int = 10
    plots: dict[str, str] = field(repr=False, default_factory=dict)
    test_name: str = "Shapely Feature Importance Test"
    test_desc: str = None

    def __post_init__(self):
        default_test_desc = inspect.cleandoc(
            f"""
           Test if the subgroups of the protected attributes are the top ranking influential
           variables under shapely feature importance value. To pass, subgroups should not
           be ranked in the top {self.threshold} features.
        """
        )

        self.test_desc = default_test_desc if self.test_desc is None else self.test_desc

    def get_shap_values(self, model, model_type, x_train_encoded, x_test_encoded):
        if model_type == "trees":
            explainer = shap.TreeExplainer(model=model, model_output="margin", feature_perturbation="tree_path_dependent")
        elif model_type == "others":
            explainer = shap.PermutationExplainer(
                model=model.predict_proba, data=x_train_encoded
            )
        else:
            raise ValueError("model_type should be 'trees' or 'others'")

        self.shap_values = explainer.shap_values(x_test_encoded)

        return self.shap_values

    def shap_summary_plot(self, x_test_encoded, save_plots: bool = True):
        """
        Make a shap summary plot.

        :x_test_encoded: data to be used for shapely explanations, categorical features have to be encoded
        :save_plots: if True, saves the plots to the class instance
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
        self, model, model_type: str, x_train_encoded: DataFrame, x_test_encoded: DataFrame
    ) -> list:
        """
        Output the protected attributes that are listed in the top specified % of the features influencing the predictions
        , using aggregated shapely values.
        """
        if ('object' in list(x_train_encoded.dtypes)) or ('object' in list(x_test_encoded.dtypes)):
            raise AttributeError(f"Categorical features have to be encoded.")
        
        shap_values = self.get_shap_values(model, model_type, x_train_encoded, x_test_encoded)

        # Take the mean of the absolute of the shapely values to get the aggretated importance for each features
        agg_shap_df = DataFrame(
            DataFrame(shap_values[1], columns=x_test_encoded.columns).abs().mean()
        ).sort_values(0, ascending=False)
        agg_shap_df["feature_rank"] = agg_shap_df[0].rank(ascending=False)
        agg_shap_df.drop(0, axis=1, inplace=True)
        attrs_string = "|".join([f"{x}_" for x in self.attrs])
        result = agg_shap_df[agg_shap_df.index.to_series().str.contains(attrs_string)].copy()
        result["passed"] = result.feature_rank.apply(
            lambda x: True if x > self.threshold else False
        )

        return result

    def shap_dependence_plot(self, x_test_encoded, show_all: bool = True, save_plots: bool = True):
        """
        Create a SHAP dependence plot to show the significant effect of the flagged
        protected attributes across the whole dataset.
        
        :show_all: if false, only show attributes that failed the test
        """
        if self.result is None:
            raise AttributeError("Cannot create dependence plot before running test.")
        if show_all:
            attrs_to_show=self.result.index
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
        self, model, model_type: str, x_train_encoded: DataFrame, x_test_encoded: DataFrame
    ) -> bool:
        """
        Runs test by calculating result and evaluating if it passes a defined condition.

        :model: trained model object
        :model_type: type of algorithim, choose from ['trees','others']
        :x_train_encoded: training data features, categorical features have to be encoded
        :x_test_encoded: data to be used for shapely explanations, preferably eval set, categorical features have to be encoded
        """
        self.result = self.get_result(model, model_type, x_train_encoded, x_test_encoded)
        self.passed = False if False in list(self.result.passed) else True

        return self.passed
