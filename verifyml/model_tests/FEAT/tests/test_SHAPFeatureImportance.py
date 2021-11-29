# Test cases for the SHAPFeatureImportance FEAT test

from ..SHAPFeatureImportance import SHAPFeatureImportance

import inspect


# Only testing the default plot inputs here since test outputs are determined by shapely library functions and require model object input
def test_plot_defaults():
    """Test that the default arguments of the plot() method are as expected."""

    sig = inspect.signature(SHAPFeatureImportance.shap_summary_plot)
    sig2 = inspect.signature(SHAPFeatureImportance.shap_dependence_plot)

    assert sig.parameters["save_plots"].default == True

    assert sig2.parameters["save_plots"].default == True
    assert sig2.parameters["show_all"].default == True
