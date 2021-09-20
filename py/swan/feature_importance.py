import pandas as pd
import matplotlib.pyplot as plt

def pa_in_top_feature_importance(protected_attr:list, top_n:int, df_importance:pd.DataFrame):
    '''
    Output the protected attributes that are listed in the top specified number of the features, 
    using feature importance values inputted by the user.
    
    :protected_attr: list of protected attributes
    :top_n: the top n features to be specified
    :df_importance: A dataframe with 2 columns - first column of feature names and second column of importance values
    
    '''
    # Sort the df by most important features, descending order
    df_importance = df_importance.sort_values(df_importance.columns[1],ascending=False)
    
    # Plot top 10 important features
    plt.figure(figsize=(15,8))
    plt.bar(df_importance.iloc[:10,0], df_importance.iloc[:10,1])
    plt.title('Feature Importance Plot')
    plt.ylabel('Relative Importance Value')
    plt.show()
    
    top_feats=df_importance.iloc[:top_n,0]
    result=[]
    for i in protected_attr:
        result = result + [j for j in top_feats if i+'_' in j ]
    return result