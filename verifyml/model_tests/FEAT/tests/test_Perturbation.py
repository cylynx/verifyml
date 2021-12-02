# Test cases for the Perturbation FEAT test

from ..Perturbation import Perturbation
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

import inspect
import category_encoders as ce
import pandas as pd
import numpy as np


# Sample test case datas
x_train_data = pd.DataFrame(
    {
        "gender": ["M", "F", "F", "M"],
        "income": [100, 200, 150, 400],
    }
)

y_train_data = [0, 1, 0, 1]

x_test_data = pd.DataFrame(
    {
        "gender": ["M", "M", "M", "M", "M", "F", "F", "F", "F", "F"],
        "income": [100, 200, 150, 400, 600, 800, 900, 500, 50, 200],
    }
)
y_test_data = [1, 0, 0, 1, 0, 1, 1, 1, 1, 0]


# Train simple randomforest model with categorical encoding
estimator = Pipeline(
    steps=[
        ("onehot", ce.OneHotEncoder(use_cat_names=True)),
        ("classifier", RandomForestClassifier(random_state=882)),
    ]
)

estimator.fit(x_train_data, y_train_data)


# Only testing the default plot inputs here since test outputs are determined by random perturbation of feature and require model object input
def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(Perturbation.plot)

    assert sig.parameters["alpha"].default == 0.05
    assert sig.parameters["save_plots"].default == True


def test_save_plots_true():
    """Test that the plot is saved to the test object when .plot(save_plots=True)."""
    # init test object
    pmt_test = Perturbation(
        attr="gender",
        metric="fpr",
        method="ratio",
        threshold=1.5,
    )

    # run test
    pmt_test.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    # plot it
    pmt_test.plot(save_plots=True)

    # test object should be a dict of length 1
    assert len(pmt_test.plots) == 1

    # test object should have the specified key, and the value should be a string
    assert isinstance(
        pmt_test.plots["False Positive Rates across gender subgroups"], str
    )


def test_save_plots_false():
    """Test that the plot is not saved to the test object when .plot(save_plots=False)."""
    # init test object
    pmt_test = Perturbation(
        attr="gender",
        metric="fpr",
        method="ratio",
        threshold=1.5,
    )

    # run test
    pmt_test.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    # plot it
    pmt_test.plot(save_plots=False)

    # nothing should be saved
    assert len(pmt_test.plots) == 0


def test_run_ratio():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test object
    pmt_test1 = Perturbation(
        attr="gender",
        metric="fpr",
        method="ratio",
        threshold=1.5,
    )

    pmt_test2 = Perturbation(
        attr="gender",
        metric="fnr",
        method="ratio",
        threshold=1.5,
    )

    pmt_test3 = Perturbation(
        attr="gender",
        metric="mse",
        method="ratio",
        threshold=1.5,
    )

    pmt_test4 = Perturbation(
        attr="gender",
        metric="mae",
        method="ratio",
        threshold=0.9,
    )

    # run test
    np.random.seed(1235)
    pmt_test1.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    np.random.seed(1235)
    pmt_test2.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    np.random.seed(1235)
    pmt_test3.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    np.random.seed(1235)
    pmt_test4.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )
    assert pmt_test1.result.loc["gender_F", "ratio"] == 2
    assert pmt_test1.result.loc["gender_M", "ratio"] == 0.667  # rounded to 3 d.p
    assert pmt_test1.passed == False

    assert pmt_test2.result.loc["gender_F", "ratio"] == 0.75
    assert pmt_test2.result.loc["gender_M", "ratio"] == 1.5
    assert pmt_test2.passed == True

    assert pmt_test3.result.loc["gender_F", "ratio"] == 1
    assert pmt_test3.result.loc["gender_M", "ratio"] == 1
    assert pmt_test3.passed == True

    assert pmt_test4.result.loc["gender_F", "ratio"] == 1
    assert pmt_test4.result.loc["gender_M", "ratio"] == 1
    assert pmt_test4.passed == False


def test_run_diff():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    # init test object
    pmt_test1 = Perturbation(
        attr="gender",
        metric="fpr",
        method="diff",
        threshold=0.35,
    )

    pmt_test2 = Perturbation(
        attr="gender",
        metric="fnr",
        method="diff",
        threshold=0.35,
    )

    pmt_test3 = Perturbation(
        attr="gender",
        metric="mse",
        method="diff",
        threshold=0.5,
    )

    pmt_test4 = Perturbation(
        attr="gender",
        metric="mae",
        method="diff",
        threshold=-1,
    )

    # run test
    np.random.seed(1235)
    pmt_test1.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    np.random.seed(1235)
    pmt_test2.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    np.random.seed(1235)
    pmt_test3.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    np.random.seed(1235)
    pmt_test4.run(
        x_test=x_test_data,
        y_test=y_test_data,
        encoder=estimator[0],
        model=estimator[-1],
    )

    assert pmt_test1.result.loc["gender_F", "difference"] == 0.5
    assert pmt_test1.result.loc["gender_M", "difference"] == -0.167  # rounded to 3 d.p
    assert pmt_test1.passed == False

    assert pmt_test2.result.loc["gender_F", "difference"] == -0.083  # rounded to 3 d.p
    assert pmt_test2.result.loc["gender_M", "difference"] == 0.167  # rounded to 3 d.p
    assert pmt_test2.passed == True

    assert pmt_test3.result.loc["gender_F", "difference"] == 0
    assert pmt_test3.result.loc["gender_M", "difference"] == 0
    assert pmt_test3.passed == True

    assert pmt_test4.result.loc["gender_F", "difference"] == 0
    assert pmt_test4.result.loc["gender_M", "difference"] == 0
    assert pmt_test4.passed == False
