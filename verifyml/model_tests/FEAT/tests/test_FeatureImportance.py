# Test cases for the FeatureImportance FEAT test

from ..FeatureImportance import FeatureImportance

import inspect
import pandas as pd

# Read test case data
test_data = pd.DataFrame(
    {
        "features": [
            "income",
            "gender_M",
            "gender_F",
            "married_No",
            "amt",
            "age",
            "married_Yes",
        ],
        "value": [0.7, 0.4, 0.2, 0.1, 0.6, 0.5, 0.3],
    }
)


def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(FeatureImportance.plot)

    assert sig.parameters["show_n"].default == 10
    assert sig.parameters["save_plots"].default == True


def test_save_plots_true():
    """Test that the plot is saved to the test object when .plot(save_plots=True)."""
    # init test object
    test_obj = imp_test = FeatureImportance(attrs=["gender", "married"], threshold=4)

    # plot it
    test_obj.plot(test_data, save_plots=True)

    # test object should be a dict of length 1
    assert len(test_obj.plots) == 1

    # test object should have the specified key, and the value should be a string
    assert isinstance(test_obj.plots["Feature Importance Plot"], str)


def test_save_plots_false():
    """Test that the plot is not saved to the test object when .plot(save_plots=False)."""
    # init test object
    test_obj = imp_test = FeatureImportance(attrs=["gender", "married"], threshold=4)

    # plot it
    test_obj.plot(test_data, save_plots=False)

    # nothing should be saved
    assert len(test_obj.plots) == 0


def test_run():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test object
    test_obj = imp_test = FeatureImportance(attrs=["gender", "married"], threshold=4)

    # run test
    test_obj.run(test_data)

    test_obj.result.loc["gender_M"].feature_rank == 4
    test_obj.result.loc["gender_F"].feature_rank == 6
    test_obj.result.loc["married_Yes"].feature_rank == 5
    test_obj.result.loc["married_No"].feature_rank == 7

    test_obj.result.loc["gender_M"].passed == False
    test_obj.result.loc["gender_F"].passed == True
    test_obj.result.loc["married_Yes"].passed == True
    test_obj.result.loc["married_No"].passed == True

    assert test_obj.passed == False
