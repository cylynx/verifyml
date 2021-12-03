# Test cases for the SHAPFeatureImportance FEAT test

from ..SHAPFeatureImportance import SHAPFeatureImportance
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import inspect
import category_encoders as ce
import pandas as pd

# Sample test case data
x_test_data = pd.DataFrame(
    {
        "gender": ["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"],
        "married": ["Y", "Y", "Y", "N", "N", "Y", "Y", "N", "N", "N"],
        "income": [100, 200, 150, 400, 600, 800, 900, 500, 50, 200],
    }
)

y_test_data = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]


# Train simple randomforest model with categorical encoding
estimator = Pipeline(
    steps=[
        ("onehot", ce.OneHotEncoder(use_cat_names=True)),
        ("classifier", RandomForestClassifier(random_state=882)),
    ]
)

estimator.fit(x_test_data, y_test_data)


# Only testing the default plot inputs here since test outputs are determined by shapely library functions and require model object input
def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(SHAPFeatureImportance.shap_summary_plot)
    sig2 = inspect.signature(SHAPFeatureImportance.shap_dependence_plot)

    assert sig.parameters["save_plots"].default == True

    assert sig2.parameters["save_plots"].default == True
    assert sig2.parameters["show_all"].default == True


def test_save_plots_true():
    """Test that the plot is saved to the test object when .plot(save_plots=True)."""
    # init test object
    shap_test = SHAPFeatureImportance(attrs=["gender", "married"], threshold=3)

    # run test
    shap_test.run(
        model=estimator[-1],
        model_type="trees",
        x_train_encoded=estimator[0].transform(x_test_data),
        x_test_encoded=estimator[0].transform(x_test_data),
    )

    # plot it
    shap_test.shap_summary_plot(estimator[0].transform(x_test_data), save_plots=True)
    shap_test.shap_dependence_plot(
        estimator[0].transform(x_test_data), save_plots=True, show_all=False
    )

    # test object should be a dict of length 3
    assert len(shap_test.plots) == 3

    # test object should have the specified key, and the value should be a string
    assert isinstance(shap_test.plots["SHAP Summary Plot"], str)
    assert isinstance(shap_test.plots["SHAP Dependence Plot: gender_M"], str)
    assert isinstance(shap_test.plots["SHAP Dependence Plot: married_N"], str)


def test_save_plots_false():
    """Test that the plot is not saved to the test object when .plot(save_plots=False)."""
    # init test object
    shap_test = SHAPFeatureImportance(attrs=["gender", "married"], threshold=3)

    # run test
    shap_test.run(
        model=estimator[-1],
        model_type="trees",
        x_train_encoded=estimator[0].transform(x_test_data),
        x_test_encoded=estimator[0].transform(x_test_data),
    )

    # plot it
    shap_test.shap_summary_plot(estimator[0].transform(x_test_data), save_plots=False)
    shap_test.shap_dependence_plot(
        estimator[0].transform(x_test_data), save_plots=False, show_all=False
    )

    # nothing should be saved
    assert len(shap_test.plots) == 0


def test_run():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test objects
    shap_test = SHAPFeatureImportance(attrs=["gender", "married"], threshold=3)

    # run test
    shap_test.run(
        model=estimator[-1],
        model_type="trees",
        x_train_encoded=estimator[0].transform(x_test_data),
        x_test_encoded=estimator[0].transform(x_test_data),
    )

    assert shap_test.result.loc["gender_M", "feature_rank"] == 2
    assert shap_test.result.loc["gender_F", "feature_rank"] == 4
    assert shap_test.result.loc["married_N", "feature_rank"] == 3
    assert shap_test.result.loc["married_Y", "feature_rank"] == 5
    assert shap_test.passed == False
