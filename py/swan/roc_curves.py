import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def roc_curve_groups_test(
    attr: str, 
    df_test_with_output: pd.DataFrame, 
    metric:str = 'tpr' , 
    metric_threshold : int = 0.7, 
    proba_thresholds = 0.5
):
    '''
    Test if at the current probability thresholds, for a particular attribute, the fpr/tpr of its groups 
    passes the maximum/mininum specified metric thresholds. Output the list of groups which fails the test. 
    Also, generate roc curves for each groups of a protected attribute, plot all on the same chart.
    The maximum number of groups allowed in a protected attribute is 10.
    Mark the optimal probability thresholds (maximise tpr-fpr) and the 
    specified current probability thresholds on the chart, default 0.5
    
    :attr: protected attribute, e.g. 'gender'
    :df_test_with_output: evaluation set dataframe containing protected attributes with "prediction_probas" and "truth" columns,
                          protected attribute should not be encoded yet
    :metric: choose from ['fpr','tpr']
    :metric_threshold: To pass, fpr has to be lower than the threshold or tpr has to be greater than the thresholds specified
    :proba_thresholds: optional argument. dictionary object with keys as the attribute groups and the values as the thresholds 
                       for the output to be classified as 1, default input will set thresholds of each group to be 0.5
    '''
    
    # To allocate different coloring for different attribute group
    colour_lst = ["red", "blue", "grey", "green","black","brown","purple","orange","magenta","pink"]
    result=[]
    plt.figure(figsize=(13,6))
    #attr_lst= [i for i in x_test.columns if attr in i]
    for i in df_test_with_output[attr].unique():
        color = colour_lst.pop(0)
        output_sub=df_test_with_output[df_test_with_output[attr]==i]
        fpr, tpr, thresholds_lst = roc_curve(output_sub['truth'],  output_sub['prediction_probas'])
        
        if type(proba_thresholds) == dict:
            threshold = proba_thresholds[i]
        else:   # if threshold dict is not specified, show the markers for default probability threshold = 0.5
            threshold = 0.5
        tmp=[i for i in thresholds_lst-threshold if i>0]
        idx=tmp.index(tmp[-1])
        if (metric == 'fpr') and (fpr[idx] > metric_threshold):
            result.append(i)
        elif (metric == 'tpr') and (tpr[idx] < metric_threshold):
            result.append(i)
            
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds_lst[optimal_idx+1]
        optimal_txt='Optimal Prob Threshold'
        txt='Current Prob Threshold'
        plt.scatter(fpr[optimal_idx],tpr[optimal_idx],color=color, marker ='.',s=70, label=optimal_txt+ ' = '+str(optimal_threshold)+ ', '+attr+'_'+i)
        plt.scatter(fpr[idx],tpr[idx],color=color, marker ='x',s=30, label=txt+ ' = '+str(threshold)+ ', '+attr+'_'+i)
        plt.plot(fpr, tpr, label='ROC of '+attr+'_'+i,color=color)
    plt.xlabel("False Positive Rate", fontsize=15)
    plt.ylabel("True Positive Rate", fontsize=15)
    #plt.plot([0,1], [0,1], color='black', linestyle='--')
    if metric == 'tpr':
        plt.axhline(y=metric_threshold, color='black', linestyle='--', label= 'Mininum TPR Threshold = '+str(metric_threshold))
    elif metric == 'fpr':
        plt.axvline(x=metric_threshold, color='black', linestyle='--', label= 'Maximum FPR Threshold = '+str(metric_threshold))
    plt.title('ROC Curve of '+attr+ ' groups', fontsize=15)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    return result