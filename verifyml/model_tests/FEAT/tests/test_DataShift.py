# Test cases for the DataShift FEAT test

from ..DataShift import DataShift

import inspect

# TODO: create a simple set of data to be used as a test case
test_data = ...


def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(DataShift.plot)

    assert sig.parameters["alpha"].default == 0.05
    assert sig.parameters["save_plots"].default == True


'''
# TODO (included a suggested approach)
def test_save_plots_true():
    """Test that the plot is saved to the test object when .plot(save_plots=True)."""
    # init test object
    test_obj = DataShift()

    # read test data
    ...

    # plot it
    test_obj.plot(save_plots=True)

    # test object should be a dict of length 1
    assert len(test_obj.plots) == 1

    # test object should have the specified key, and the value should be a string
    assert isinstance(test_obj.plots["Probability Distribution of protected attributes"], str)

    # other assertions
    ...


# TODO (included a suggested approach)
def test_save_plots_false():
    """Test that the plot is not saved to the test object when .plot(save_plots=False)."""

    test_obj = DataShift()
    ...  # read test data here
    test_obj.plot(save_plots=False)

    # nothing should be saved
    assert test_obj.plots is None

    # other assertions
    ...


# TODO
def test_run():
    """Test that calling .run() updates the test object's .result and .passed attributes."""
    test_obj = DataShift()
    ...  # read test data here
    test_obj.run()

    assert test_obj.result == ...
    assert isinstance(test_obj.passed, bool)


# other tests
...
'''
