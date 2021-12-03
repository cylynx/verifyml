# Test cases for the DataShift FEAT test

from ..DataShift import DataShift

import inspect
import pandas as pd

# Sample test case datas
x_train_data = pd.DataFrame(
    {"gender": ["M", "M", "M", "M", "M", "M", "F", "F", "F", "F"]}
)
x_test_data = pd.DataFrame(
    {"gender": ["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"]}
)


def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(DataShift.plot)

    assert sig.parameters["alpha"].default == 0.05
    assert sig.parameters["save_plots"].default == True


def test_save_plots_true():
    """Test that the plot is saved to the test object when .plot(save_plots=True)."""
    # init test object
    data_test = DataShift(protected_attr=["gender"], method="ratio", threshold=1.5)

    # run test
    data_test.run(x_train=x_train_data, x_test=x_test_data)

    # plot it
    data_test.plot(save_plots=True)

    # test object should be a dict of length 1
    assert len(data_test.plots) == 1

    # test object should have the specified key, and the value should be a string
    assert isinstance(
        data_test.plots["Probability Distribution of protected attributes"], str
    )


def test_save_plots_false():
    """Test that the plot is not saved to the test object when .plot(save_plots=False)."""
    # init test object
    data_test = DataShift(protected_attr=["gender"], method="ratio", threshold=1.5)

    # run test
    data_test.run(x_train=x_train_data, x_test=x_test_data)

    # plot it
    data_test.plot(save_plots=False)

    # nothing should be saved
    assert len(data_test.plots) == 0


def test_run_ratio():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test object
    data_test = DataShift(protected_attr=["gender"], method="ratio", threshold=1.23)

    # run test
    data_test.run(x_train=x_train_data, x_test=x_test_data)

    assert data_test.result.loc["gender_F"]["training_distribution"] == 0.4
    assert data_test.result.loc["gender_F"]["eval_distribution"] == 0.5
    assert data_test.result.loc["gender_F"]["ratio"] == 1.25
    assert data_test.result.loc["gender_F"]["passed"] == False

    assert data_test.result.loc["gender_M"]["training_distribution"] == 0.6
    assert data_test.result.loc["gender_M"]["eval_distribution"] == 0.5
    assert data_test.result.loc["gender_M"]["ratio"] == 1.2
    assert data_test.result.loc["gender_M"]["passed"] == True

    assert data_test.passed == False


def test_run_difference():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test object
    data_test = DataShift(protected_attr=["gender"], method="diff", threshold=0.1)

    # run test
    data_test.run(x_train=x_train_data, x_test=x_test_data)

    assert data_test.result.loc["gender_F"]["training_distribution"] == 0.4
    assert data_test.result.loc["gender_F"]["eval_distribution"] == 0.5
    assert data_test.result.loc["gender_F"]["difference"] == 0.1
    assert data_test.result.loc["gender_F"]["passed"] == True

    assert data_test.result.loc["gender_M"]["training_distribution"] == 0.6
    assert data_test.result.loc["gender_M"]["eval_distribution"] == 0.5
    assert data_test.result.loc["gender_M"]["difference"] == 0.1
    assert data_test.result.loc["gender_M"]["passed"] == True

    assert data_test.passed == True


def test_run_chi2():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test object
    data_test = DataShift(protected_attr=["gender"], method="chi2", threshold=1.1)

    # run test
    data_test.run(x_train=x_train_data, x_test=x_test_data)

    assert data_test.result.loc["gender_F"]["training_distribution"] == 0.4
    assert data_test.result.loc["gender_F"]["eval_distribution"] == 0.5
    assert data_test.result.loc["gender_F"]["p-value"] == 1
    assert data_test.result.loc["gender_F"]["passed"] == False

    assert data_test.result.loc["gender_M"]["training_distribution"] == 0.6
    assert data_test.result.loc["gender_M"]["eval_distribution"] == 0.5
    assert data_test.result.loc["gender_M"]["p-value"] == 1
    assert data_test.result.loc["gender_M"]["passed"] == False

    assert data_test.passed == False
