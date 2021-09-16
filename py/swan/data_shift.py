import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_shift_test(
    protected_attr: list, 
    threshold: int,
    df_train: pd.DataFrame, 
    df_eval: pd.DataFrame, 
):
    '''
    Test if there is any shift (based on specified threshold) in the distribution of the protected feature, 
    which may impose new unfairness and require a retraining of the model, output the shifted attributes 
    
    :protected_attr: list of protected attributes
    :threshold: probability distribution threshold of an attribute, where if the difference between training data
                     distribution and evalaution distribution exceeds the threhold, the attribute will be flagged
    :df_train: training data features, protected features should not be encoded yet
    :df_eval: data to be evaluated on, protected features should not be encoded yet
    '''
    result=[]
    fig, axs = plt.subplots(1, len(protected_attr), figsize=(15, 4),)
    num=0
    for pa in protected_attr:
        df_train=df_train.sort_values(pa)
        sns.histplot(x=pa,  data=df_train, ax=axs[num],stat="probability",color='green')
        num+=1
    fig.suptitle('Distribution of protected attributes in training set')
    plt.show()
    
    fig, axs = plt.subplots(1, len(protected_attr), figsize=(15, 4),)
    num=0
    for pa in protected_attr:
        df_eval=df_eval.sort_values(pa)
        sns.histplot(x=pa,  data=df_eval, ax=axs[num],stat="probability", color='red')
        num+=1
    fig.suptitle('Distribution of protected attributes in test set')
    plt.show()
    
    for pa in protected_attr:
        train_dist = df_train.groupby(pa)[pa].apply(lambda x: x.count()/len(df_train))
        eval_dist = df_eval.groupby(pa)[pa].apply(lambda x: x.count()/len(df_eval))
        if sum(abs(train_dist - eval_dist) > threshold):
            result.append(pa)
    return result