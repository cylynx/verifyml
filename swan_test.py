import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from py.swan.feature_importance import pa_in_top_feature_importance
from py.swan.feature_importance_shap import pa_in_top_feature_importance_shap
from py.swan.data_shift import data_shift_test
from py.swan.bias_metrics import generate_bias_metrics_charts, bias_metrics_test
from py.swan.perturbation import bias_metrics_permutation_test
from py.swan.roc_curves import roc_curve_groups_test
from py.FEATTests import (
    DataShift,
    SubgroupDifference,
    Permutation,
    FeatureImportance,
    SHAPFeatureImportance,
)

df=pd.read_csv('data/creditcard.csv',nrows=100000).drop('Time',axis=1)
df=df[['V1','V2','V3','V4','V5','Class']]
df=pd.concat([df[df.Class==0].sample(2000), df[df.Class==1]]).reset_index(drop=True)
# df=pd.read_csv('syn.csv',nrows=100000)
# df=df[['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFraud']]

# Add mocked protected attributes columns
df['age']=df.V1.apply(lambda x: np.random.choice(["<=17", "18-25", "26-39", ">=40"], p=[0.1, 0.3,0.3,0.3]))
df['gender']=df.V1.apply(lambda x: np.random.choice(["M", "F"], p=[0.5, 0.5]))

x=df.drop('Class',axis=1)
y=df['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=32)
estimator = RandomForestClassifier(n_estimators=10, max_features='sqrt')
#estimator = LogisticRegression()

#Apply one hot encoding to categorical columns (auto-detect object columns)
ens=ce.OneHotEncoder(use_cat_names=True)
x_train=ens.fit_transform(x_train)
x_test=ens.transform(x_test)

estimator.fit(x_train, y_train)

output=x_test.copy()
y_pred = estimator.predict(x_test)
y_probas = estimator.predict_proba(x_test)[::,1]
print(confusion_matrix(y_test, y_pred))

output=ens.inverse_transform(output)
output['truth']=y_test
output['prediction']=y_pred
output['prediction_probas']=y_probas


df_importance = pd.DataFrame({'features':x_test.columns,'value':estimator.feature_importances_})

# result = pa_in_top_feature_importance(
#     protected_attr=['gender','age'],
#     top_n=6,
#     df_importance=df_importance,
# )

# result = FeatureImportance(
#     test_name='my feature importance FEAT test',
#     test_desc='',
#     attrs=['gender','age'],
#     top_n=6
# )

# result.run(df_importance)

# result = pa_in_top_feature_importance_shap(
#     protected_attr=['gender','age'],
#     top_n=7,
#     model=estimator,
#     model_type='trees',
#     x_train=x_train,
#     x_test=x_test
# )

result = SHAPFeatureImportance(
    test_name='SHAP',
    test_desc='',
    attrs=['gender','age'],
    top_n=7
)

result.run(
    model=estimator,
    model_type='trees',
    x_train=x_train,
    x_test=x_test
)

# train=ens.inverse_transform(x_train)
# test=ens.inverse_transform(x_test)

# result = data_shift_test(
#     protected_attr = ['gender','age'],
#     threshold = 0.05,
#     df_train = train,
#     df_eval = test 
# )

# result = DataShift(
#     test_name='my data shift FEAT test',
#     test_desc='',
#     protected_attr = ['gender','age'],
#     threshold = 0.05
# )

# result.run(df_train = train, df_eval = test)

# result=generate_bias_metrics_charts(
#                 protected_attr = ['gender','age'],
#                 df_test_with_output = output
#             )

# result = bias_metrics_test(
#     attr='gender',
#     metric='sr',
#     method='ratio',
#     threshold=1.5,
#     df_test_with_output = output
# )

# result = SubgroupDifference(
#     test_name='subgroup diff',
#     test_desc='',
#     attr='gender',
#     metric='sr',
#     method='ratio',
#     threshold=1.5,
# )

# result.run(df_test_with_output=output)


# result =bias_metrics_permutation_test(
#     attr='age',
#     metric='sr',
#     method='ratio',
#     threshold=1.25,
#     x_test=x_test,
#     y_test=y_test,
#     encoder=ens,
#     model=estimator
# )

# result = Permutation(
#     test_name='permutation',
#     test_desc='',
#     attr='age',
#     metric='sr',
#     method='ratio',
#     threshold=1.25,
# )

# result.run(
#     x_test=x_test,
#     y_test=y_test,
#     encoder=ens,
#     model=estimator
# )

# result = roc_curve_groups_test(
#     attr = 'age',
#     df_test_with_output = output,
#     metric = 'tpr',
#     metric_threshold = 0.65,
#     #proba_thresholds = {'<=17':0.5,'>=40':0.6,'18-25':0.4,'26-39':0.3}
# )

# print(result.__dict__)
print(result)