# Test cases for the MinMaxMetricThreshold FEAT test

from ..MinMaxMetricThreshold import MinMaxMetricThreshold

import inspect
import pandas as pd

# Read test case data
test_data_classification = pd.DataFrame(
    {
        "gender": ["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"],
        "truth": [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        "prediction_probas": [0.6, 0.4, 0.5, 0.6, 0.2, 0.7, 0.1, 1, 0.2, 0.3],
    }
)

test_data_regression = pd.DataFrame(
    {
        "gender": ["M", "M", "F", "F"],
        "truth": [1, 0, 1, 1],
        "prediction": [1, 1, 1, 0.8],
    }
)


def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(MinMaxMetricThreshold.plot)

    assert sig.parameters["alpha"].default == 0.05
    assert sig.parameters["save_plots"].default == True


def test_save_plots_true_classification():
    """Test that the plot is saved to the test object when .plot(save_plots=True)."""
    # init test object
    test_obj = MinMaxMetricThreshold(attr="gender", metric="fpr", threshold=0.025,)

    # run test
    test_obj.run(test_data_classification)

    # plot it
    test_obj.plot(save_plots=True)

    # test object should be a dict of length 1
    assert len(test_obj.plots) == 1

    # test object should have the specified key, and the value should be a string
    assert isinstance(test_obj.plots["ROC Curve of gender groups"], str)


def test_save_plots_true_regression():
    """Test that the plot is saved to the test object when .plot(save_plots=True)."""
    # init test object
    test_obj = MinMaxMetricThreshold(attr="gender", metric="mse", threshold=0.5,)

    # run test
    test_obj.run(test_data_regression)

    # plot it
    test_obj.plot(save_plots=True)

    # test object should be a dict of length 1
    assert len(test_obj.plots) == 1

    # test object should have the specified key, and the value should be a string
    assert isinstance(test_obj.plots["Mean Squared Error across gender subgroups"], str)


def test_save_plots_false_classification():
    """Test that the plot is not saved to the test object when .plot(save_plots=False)."""
    # init test object
    test_obj = MinMaxMetricThreshold(attr="gender", metric="fnr", threshold=0.5,)

    # run test
    test_obj.run(test_data_classification)

    # plot it
    test_obj.plot(save_plots=False)

    # nothing should be saved
    assert len(test_obj.plots) == 0


def test_save_plots_false_regression():
    """Test that the plot is not saved to the test object when .plot(save_plots=False)."""
    # init test object
    test_obj = MinMaxMetricThreshold(attr="gender", metric="mae", threshold=0.5,)

    # run test
    test_obj.run(test_data_regression)

    # plot it
    test_obj.plot(save_plots=False)

    # nothing should be saved
    assert len(test_obj.plots) == 0


def test_run_classification():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test object
    test_obj1 = MinMaxMetricThreshold(attr="gender", metric="fpr", threshold=0.5,)

    test_obj2 = MinMaxMetricThreshold(attr="gender", metric="fnr", threshold=0.7,)

    # run tests
    test_obj1.run(test_data_classification)
    test_obj2.run(test_data_classification)

    assert (
        test_obj1.result.loc["gender_F"]["fpr at current probability threshold"]
        == 0.333
    )  # rounded to 3 d.p
    assert (
        test_obj1.result.loc["gender_M"]["fpr at current probability threshold"] == 0.5
    )
    assert test_obj1.passed == False

    assert (
        test_obj2.result.loc["gender_F"]["fnr at current probability threshold"] == 0.5
    )
    assert (
        test_obj2.result.loc["gender_M"]["fnr at current probability threshold"]
        == 0.667
    )  # rounded to 3 d.p
    assert test_obj2.passed == True


def test_run_regression():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test object
    test_obj1 = MinMaxMetricThreshold(attr="gender", metric="mse", threshold=0.51,)

    test_obj2 = MinMaxMetricThreshold(attr="gender", metric="mae", threshold=0.4,)

    # run tests
    test_obj1.run(test_data_regression)
    test_obj2.run(test_data_regression)

    assert test_obj1.result.loc["gender_F"]["mse"] == 0.02  # rounded to 3 d.p
    assert test_obj1.result.loc["gender_M"]["mse"] == 0.5
    assert test_obj1.passed == True

    assert test_obj2.result.loc["gender_F"]["mae"] == 0.1
    assert test_obj2.result.loc["gender_M"]["mae"] == 0.5
    assert test_obj2.passed == False
