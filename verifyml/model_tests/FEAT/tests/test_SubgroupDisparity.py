# Test cases for the SubgroupDisparity FEAT test

from ..SubgroupDisparity import SubgroupDisparity

import inspect
import pandas as pd

# Read test case data
test_data = pd.DataFrame(
    {
        "gender": ["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"],
        "truth": [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        "prediction": [1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    }
)


def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(SubgroupDisparity.plot)

    assert sig.parameters["alpha"].default == 0.05
    assert sig.parameters["save_plots"].default == True


def test_save_plots_true():
    """Test that the plot is saved to the test object when .plot(save_plots=True)."""
    # init test object
    test_obj1 = SubgroupDisparity(
        attr="gender", metric="fpr", method="ratio", threshold=1.5,
    )

    test_obj2 = SubgroupDisparity(
        attr="gender", metric="mse", method="ratio", threshold=1.5,
    )

    # run test
    test_obj1.run(test_data)
    test_obj2.run(test_data)

    # plot it
    test_obj1.plot(alpha=0.05, save_plots=True)
    test_obj2.plot(alpha=0.05, save_plots=True)

    # test object should be a dict of length 1
    assert len(test_obj1.plots) == 1
    assert len(test_obj2.plots) == 1

    # test object should have the specified key, and the value should be a string
    assert isinstance(
        test_obj1.plots["False Positive Rates across gender subgroups"], str
    )
    assert isinstance(
        test_obj2.plots["Mean Squared Error across gender subgroups"], str
    )


def test_save_plots_false():
    """Test that the plot is not saved to the test object when .plot(save_plots=False)."""
    # init test object
    test_obj1 = SubgroupDisparity(
        attr="gender", metric="fpr", method="ratio", threshold=1.5,
    )

    test_obj2 = SubgroupDisparity(
        attr="gender", metric="mse", method="ratio", threshold=1.5,
    )

    # run test
    test_obj1.run(test_data)
    test_obj2.run(test_data)

    # plot it
    test_obj1.plot(save_plots=False)
    test_obj2.plot(save_plots=False)

    # nothing should be saved
    assert len(test_obj1.plots) == 0
    assert len(test_obj2.plots) == 0


def test_run_ratio():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test objects
    test_obj1 = SubgroupDisparity(
        attr="gender", metric="fpr", method="ratio", threshold=1.4,
    )

    test_obj2 = SubgroupDisparity(
        attr="gender", metric="fnr", method="ratio", threshold=1.5,
    )

    test_obj3 = SubgroupDisparity(
        attr="gender", metric="mse", method="ratio", threshold=1.5,
    )

    test_obj4 = SubgroupDisparity(
        attr="gender", metric="mae", method="ratio", threshold=1.5,
    )

    # run tests
    test_obj1.run(test_data)
    test_obj2.run(test_data)
    test_obj3.run(test_data)
    test_obj4.run(test_data)

    assert test_obj1.result.iloc[0].gender_fpr_max_ratio == 1.5
    assert test_obj1.passed == False

    assert test_obj2.result.iloc[0].gender_fnr_max_ratio == 1.333  # rounded to 3 d.p
    assert test_obj2.passed == True

    assert test_obj3.result.iloc[0].gender_mse_max_ratio == 1.5
    assert test_obj3.passed == True

    assert test_obj4.result.iloc[0].gender_mae_max_ratio == 1.5
    assert test_obj4.passed == True


def test_run_difference():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test objects
    test_obj1 = SubgroupDisparity(
        attr="gender", metric="fpr", method="diff", threshold=0.2,
    )

    test_obj2 = SubgroupDisparity(
        attr="gender", metric="fnr", method="diff", threshold=0.1,
    )

    test_obj3 = SubgroupDisparity(
        attr="gender", metric="mse", method="diff", threshold=0.1,
    )

    test_obj4 = SubgroupDisparity(
        attr="gender", metric="mae", method="diff", threshold=0.1,
    )

    # run tests
    test_obj1.run(test_data)
    test_obj2.run(test_data)
    test_obj3.run(test_data)
    test_obj4.run(test_data)

    assert test_obj1.result.iloc[0].gender_fpr_max_diff == 0.167  # rounded to 3 d.p
    assert test_obj1.passed == True

    assert test_obj2.result.iloc[0].gender_fnr_max_diff == 0.167  # rounded to 3 d.p
    assert test_obj2.passed == False

    assert test_obj3.result.iloc[0].gender_mse_max_diff == 0.2
    assert test_obj3.passed == False

    assert test_obj4.result.iloc[0].gender_mae_max_diff == 0.2
    assert test_obj4.passed == False


def test_run_chi2():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test objects
    test_obj1 = SubgroupDisparity(
        attr="gender", metric="fpr", method="chi2", threshold=0.05,
    )

    test_obj2 = SubgroupDisparity(
        attr="gender", metric="fnr", method="chi2", threshold=0.6,
    )

    # run tests
    test_obj1.run(test_data)
    test_obj2.run(test_data)

    assert test_obj1.result.iloc[0].p_value == 0.576  # rounded to 3 d.p
    assert test_obj1.passed == True

    assert test_obj2.result.iloc[0].p_value == 0.576  # rounded to 3 d.p
    assert test_obj2.passed == False
