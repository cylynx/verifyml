import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def generate_bias_metrics_charts(
    protected_attr: list,
    df_test_with_output: pd.DataFrame,
):
    """
    Generate grouped bias metrics (false positive rates, false negative rates, predicted positive rates) charts
    and output a dictionary containing information on the maximum difference/ratio of the bias metrics
    between any 2 groups within a protected attribute

    :protected_attr: list of protected attributes
    :df_test_with_output: evaluation set dataframe containing protected attributes with "prediction" and "truth" column,
                          protected features should not be encoded
    """
    result = {}
    for pa in protected_attr:
        fnr = {}
        fpr = {}
        sr = {}
        for i in df_test_with_output[pa].unique():
            tmp = df_test_with_output[df_test_with_output[pa] == i]
            cm = confusion_matrix(tmp.truth, tmp.prediction)
            fnr[i] = cm[1][0] / cm[1].sum()
            fpr[i] = cm[0][1] / cm[0].sum()
            sr[i] = cm[1].sum() / cm.sum()
            result[pa + "_fnr_max_diff"] = max(fnr.values()) - min(fnr.values())
            result[pa + "_fnr_max_ratio"] = max(fnr.values()) / min(fnr.values())
            result[pa + "_fpr_max_diff"] = max(fpr.values()) - min(fpr.values())
            result[pa + "_fpr_max_ratio"] = max(fpr.values()) / min(fpr.values())
            result[pa + "_sr_max_diff"] = max(sr.values()) - min(sr.values())
            result[pa + "_sr_max_ratio"] = max(sr.values()) / min(sr.values())

        fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=False)
        axs[0].bar(list(fnr.keys()), list(fnr.values()))
        axs[0].set_title("False Negative Rates")
        axs[1].bar(list(fpr.keys()), list(fpr.values()))
        axs[1].set_title("False Positive Rates")
        axs[2].bar(list(sr.keys()), list(sr.values()))
        axs[2].set_title("Predicted Positive Rates")
        fig.suptitle("Attribute: " + pa)
        plt.show()
    return result


def bias_metrics_test(
    attr: str,
    metric: str,
    method: str,
    threshold: int,
    df_test_with_output: pd.DataFrame,
):
    """
    Check if the maximum difference/ratio of the specified bias metric of any 2 groups within a specified protected attribute
    exceeds the threshold specified. Output false if the test failed.

    :attr: protected attribute specified
    :metric: type of bias metric for the test, choose from ('fpr', 'fnr', 'sr'),
             'fpr' - false positive rate, 'fnr' - false negative rate, 'sr': selection rate
    :method: type of method for the test, choose from ('diff', 'ratio')
    :threshold: threshold for difference/ratio of the metric
    :df_test_with_output: dataframe containing protected attributes with "prediction" and "truth" colum.n
    """
    metric_dict = generate_bias_metrics_charts(
        protected_attr=[attr], df_test_with_output=df_test_with_output
    )
    return metric_dict[attr + "_" + metric + "_max_" + method] <= threshold
