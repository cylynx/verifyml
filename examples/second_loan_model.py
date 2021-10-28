import sys
import json
from IPython import display
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

import verifyml.model_card_toolkit as mctlib
from verifyml.model_card_toolkit import model_card_pb2, ModelCard
from verifyml.model_card_toolkit.utils.tally_form import tally_form_to_mc
from verifyml.model_tests.utils import plot_to_str
from verifyml.model_tests.FEAT import (
    SubgroupDisparity,
    MinMaxMetricThreshold,
    Perturbation,
    SHAPFeatureImportance,
    FeatureImportance,
    DataShift,
)


## Read sample dataset and create ML model

# Loan repayment dataset
df = pd.read_csv("../data/loan_approval.csv")
df = df.fillna(method="backfill").dropna()
df["Loan_Status"] = df.Loan_Status.apply(
    lambda x: 1 if x == "N" else 0
)  # 1 -> Loan Rejected, 0 -> Loan Approved
x = df.drop(["Loan_Status", "Loan_ID"], axis=1)
y = df["Loan_Status"]

# Train-Test data Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=771
)


## Build ML model with protected attributes as model features

# Apply one hot encoding to categorical columns (auto-detect object columns) and random forest model in the pipeline
estimator = Pipeline(
    steps=[
        ("onehot", ce.OneHotEncoder(use_cat_names=True)),
        (
            "classifier",
            RandomForestClassifier(n_estimators=12, max_depth=10, random_state=123),
        ),
    ]
)


# Fit, predict and compute performance metrics
estimator.fit(x_train, y_train)

output = x_test.copy()  # x_test df with output columns, to be appended later
y_probas = estimator.predict_proba(x_test)[::, 1]
y_pred = (y_probas > 0.45).astype(int)
train_pred = estimator.predict_proba(x_train)[::, 1] > 0.45

precision_train = round(precision_score(y_train, train_pred), 3)
recall_train = round(recall_score(y_train, train_pred), 3)
precision_test = round(precision_score(y_test, y_pred), 3)
recall_test = round(recall_score(y_test, y_pred), 3)


# Add output columns to this dataframe, to be used as a input for feat tests
output["truth"] = y_test
output["prediction"] = y_pred
output["prediction_probas"] = y_probas


# Dataframe with categorical features encoded
x_train_encoded = estimator[0].transform(x_train)
x_test_encoded = estimator[0].transform(x_test)


# Get feature importance values
df_importance = pd.DataFrame(
    {"features": x_test_encoded.columns, "value": estimator[-1].feature_importances_}
)


## Get confusion matrix and ROC curve on train/test set

# Train set
ConfusionMatrixDisplay.from_predictions(y_train, train_pred)
confusion_matrix_train = plot_to_str()

RocCurveDisplay.from_estimator(estimator, x_train, y_train)
roc_curve_train = plot_to_str()

# Test set
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
confusion_matrix_test = plot_to_str()

RocCurveDisplay.from_estimator(estimator, x_test, y_test)
roc_curve_test = plot_to_str()


## Run some FEAT Tests on the data

# ROC/Min Max Threshold Test
tpr_threshold_test_gender = MinMaxMetricThreshold(
    # test_name="",        # Default test name and description will be used accordingly if not specified
    # test_desc="",
    attr="Gender",
    metric="tpr",
    threshold=0.45,
    proba_threshold=0.45,  # Custom probability threshold, default at 0.5
)
tpr_threshold_test_gender.run(df_test_with_output=output)
tpr_threshold_test_gender.plot()

tpr_threshold_test_married = MinMaxMetricThreshold(
    attr="Married",
    metric="tpr",
    threshold=0.45,
    proba_threshold=0.45,  # Custom probability threshold, default at 0.5
)
tpr_threshold_test_married.run(df_test_with_output=output)
tpr_threshold_test_married.plot()

# Subgroup Disparity Test

sgd_test_married = SubgroupDisparity(
    attr="Married", metric="fpr", method="ratio", threshold=1.5,
)
sgd_test_married.run(output)
sgd_test_married.plot(alpha=0.05)  # default alpha argument shows 95% C.I bands

sgd_test_gender = SubgroupDisparity(
    attr="Gender", metric="fpr", method="ratio", threshold=1.5,
)
sgd_test_gender.run(output)
sgd_test_gender.plot(alpha=0.05)  # default alpha argument shows 95% C.I bands

# User inputted Feature importance test

imp_test = FeatureImportance(attrs=["Gender", "Married"], threshold=5)

imp_test.run(df_importance)
imp_test.plot(df_importance, show_n=10)  # Show top 10 most important features


## Bootstrap model card from tally form and scaffold assets
# We can add the quantitative analysis, explainability analysis and fairness analysis sections to a bootstrap model card for convenience. In this example, we use an existing model card which we created from the tally form response. This is meant only as an example - the dataset and risk evaluation in the model card is a fictional use case.

# Convert form response to model card protobuf
pb = tally_form_to_mc("sample-form-response-loan-approval.json")

# Initialize the mct and scaffold using the existing protobuf
mct2 = mctlib.ModelCardToolkit(
    output_dir="model_card_output", file_name="loan_approval_example"
)
mc2 = mct2.scaffold_assets(proto=pb)
mc2.model_details.name = "Loan Approval Model, outcome probability threshold = 0.45"

# ## Convert test objects to a model-card-compatible format

# In[128]:


# init model card test objects
mc_tpr_threshold_test_gender = mctlib.Test()
mc_tpr_threshold_test_married = mctlib.Test()
mc_sgd_test_married = mctlib.Test()
mc_sgd_test_gender = mctlib.Test()

# assign tests to them
mc_tpr_threshold_test_gender.read_model_test(tpr_threshold_test_gender)
mc_tpr_threshold_test_married.read_model_test(tpr_threshold_test_married)
mc_sgd_test_married.read_model_test(sgd_test_married)
mc_sgd_test_gender.read_model_test(sgd_test_gender)

# Add quantitative analysis

# Create 4 PerformanceMetric to store our results
mc2.quantitative_analysis.performance_metrics = [
    mctlib.PerformanceMetric() for i in range(0, 5)
]
mc2.quantitative_analysis.performance_metrics[0].type = "Recall"
mc2.quantitative_analysis.performance_metrics[0].value = str(recall_train)
mc2.quantitative_analysis.performance_metrics[0].slice = "Training Set"

mc2.quantitative_analysis.performance_metrics[1].type = "Precision"
mc2.quantitative_analysis.performance_metrics[1].value = str(precision_train)
mc2.quantitative_analysis.performance_metrics[1].slice = "Training Set"
mc2.quantitative_analysis.performance_metrics[
    1
].graphics.description = "Confusion matrix and ROC Curve"
mc2.quantitative_analysis.performance_metrics[1].graphics.collection = [
    mctlib.Graphic(image=confusion_matrix_train),
    mctlib.Graphic(image=roc_curve_train),
]

mc2.quantitative_analysis.performance_metrics[2].type = "Recall"
mc2.quantitative_analysis.performance_metrics[2].value = str(recall_test)
mc2.quantitative_analysis.performance_metrics[2].slice = "Test Set"

mc2.quantitative_analysis.performance_metrics[3].type = "Precision"
mc2.quantitative_analysis.performance_metrics[3].value = str(precision_test)
mc2.quantitative_analysis.performance_metrics[3].slice = "Test Set"
mc2.quantitative_analysis.performance_metrics[
    3
].graphics.description = "Confusion matrix and ROC Curve"
mc2.quantitative_analysis.performance_metrics[3].graphics.collection = [
    mctlib.Graphic(image=confusion_matrix_test),
    mctlib.Graphic(image=roc_curve_test),
]
mc2.quantitative_analysis.performance_metrics[4].type = "True positive rate"
mc2.quantitative_analysis.performance_metrics[4].tests = [
    mc_tpr_threshold_test_gender,
    mc_tpr_threshold_test_married,
]


# The bootstrap template comes with two requirements on fairness analysis:
# Minimum acceptable service and Equal false positive rate
# We add the relevant tests associated with it
mc2.fairness_analysis.fairness_reports[0].tests = [
    mc_sgd_test_married,
    mc_sgd_test_gender,
]

mct2.update_model_card(mc2)
