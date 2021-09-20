import pandas as pd
import shap

def pa_in_top_feature_importance_shap(
    protected_attr: list, 
    top_n: int, 
    model, 
    model_type: str, 
    x_train: pd.DataFrame, 
    x_test: pd.DataFrame
):
    '''
    Ouput the protected attributes that are listed in the top specified % of the features influencing the predictions
    ,using aggregated shapely values.
    
    :protected_attr: list of protected attributes
    :top_n: the top n features to be specified
    :model: trained model object
    :model_type: type of algorithim, choose from ['trees','others']
    :x_train: training data features, categorical features have to be already encoded
    :x_test: data to be used for shapely explanations, preferably eval set, categorical features have to be already encoded
    
    '''
    result=[]
    if model_type == 'trees':
        explainer = shap.TreeExplainer(model = model, model_output='margin')
    elif model_type == 'others':
        explainer = shap.PermutationExplainer(model = model.predict_proba, data=x_train)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values = shap_values[1], features = x_test, max_display=20, plot_type='dot')
    
    # Take the mean of the absolute of the shapely values to get the aggretated importance for each features
    agg_shap_df=pd.DataFrame(pd.DataFrame(shap_values[1],columns=x_test.columns).abs().mean()).sort_values(0,ascending=False)
    top_feat=list(agg_shap_df.iloc[:top_n].index)
    for i in protected_attr:
        result=result+[j for j in top_feat if i+'_' in j]
    
    # create a SHAP dependence plot to show the significant effect of the flagged protected attributes across the whole dataset
    for i in result:
        shap.dependence_plot(i, shap_values=shap_values[1], features=x_test,interaction_index=None)
        
    return result