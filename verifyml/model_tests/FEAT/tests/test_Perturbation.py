# Test cases for the Perturbation FEAT test

from ..Perturbation import Perturbation

import inspect


# Only testing the default plot inputs here since test outputs are determined by random perturbation of feature and require model object input
def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(Perturbation.plot)

    assert sig.parameters["alpha"].default == 0.05
    assert sig.parameters["save_plots"].default == True
