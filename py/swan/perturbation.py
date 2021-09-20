import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def bias_metrics_permutation_test(
    attr: str, 
    metric: str,
    method: str,
    threshold: int,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    encoder
):
    '''
    Check if the difference/ratio of specified bias metric of any group within a specified protected attribute between 
    the original dataset and the perturb dataset exceeds the threshold. Output list of groups that fails the test.
    i.e. Flag male gender group if 
    abs(False positive rate of male group in original test data - False positive rate of male group in perturbed gender test data) 
    >threshold     
    For ratio, take the higher value as the numerator.
     
    :attr: protected attribute specified
    :metric: type of bias metric for the test, choose from ('fpr', 'fnr', 'sr'), 
             'fpr' - false positive rate, 'fnr' - false negative rate, 'sr': selection rate
    :method: type of method for the test, choose from ('diff', 'ratio')
    :threshold: threshold for difference/ratio of the metric 
    :x_test: dataframe containing features to be inputted into the model predictions
    :y_test: array/list/series containing the truth of x_test
    :encoder: one hot encoder object, to allow for permutation of the protected attribute
    '''
    # Obtain rates for the original data
    rates={}
    y_pred = estimator.predict(x_test)
    df_test_with_output=encoder.inverse_transform(x_test)
    df_test_with_output['prediction'] = y_pred
    df_test_with_output['truth'] = y_test
    for i in df_test_with_output[attr].unique():
        tmp=df_test_with_output[df_test_with_output[attr] == i]
        cm=confusion_matrix(tmp.truth, tmp.prediction)
        if metric == 'fpr':
            rates[i] = cm[0][1]/cm[0].sum()
        elif metric == 'fnr':
            rates[i] = cm[1][0]/cm[1].sum()
        elif metric == 'sr':
            rates[i]=cm[1].sum()/cm.sum()
    print(rates)
    
    # Obtain rates for the perturb data
    rates_perturb={}
    df_perturb=encoder.inverse_transform(x_test)
    df_perturb[attr]=np.random.permutation(df_perturb[attr].values)
    df_perturb=encoder.transform(df_perturb)
    y_pred = estimator.predict(df_perturb)
    df_perturb=encoder.inverse_transform(df_perturb)
    df_perturb['prediction']=y_pred
    df_perturb['truth']=y_test
    for i in df_perturb[attr].unique():
        tmp=df_perturb[df_perturb[attr] == i]
        cm=confusion_matrix(tmp.truth, tmp.prediction)
        if metric == 'fpr':
            rates_perturb[i] = cm[0][1]/cm[0].sum()
        elif metric == 'fnr':
            rates_perturb[i] = cm[1][0]/cm[1].sum()
        elif metric == 'sr':
            rates_perturb[i]=cm[1].sum()/cm.sum()
    print(rates_perturb)    
    
    # Comapre the rates and output the attribute groups which has failed the test
    result = []
    for i in rates.keys():
        if rates[i] > rates_perturb[i]:
            if method == 'ratio':
                val=rates[i]/rates_perturb[i]
            elif method == 'diff':
                val=rates[i]-rates_perturb[i]
        else:
            if method == 'ratio':
                val=rates_perturb[i]/rates[i]
            elif method == 'diff':
                val=rates_perturb[i]-rates[i]

        if val > threshold :
            result.append(i)
    return result


    
        