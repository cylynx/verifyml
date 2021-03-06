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

# parent directory to work with dev
sys.path.append("..")
import verifyml.model_card_toolkit as mctlib
from verifyml.model_card_toolkit import model_card_pb2, ModelCard
from verifyml.model_tests.utils import plot_to_str
from verifyml.model_tests.FEAT import (
    SubgroupDisparity,
    MinMaxMetricThreshold,
    Perturbation,
    SHAPFeatureImportance,
    FeatureImportance,
    DataShift,
)


## Prepare sample test dataset and fit a random forest model

# Credit card fraud Dataset
df = pd.read_csv("../data/fraud.csv")
x = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.87, random_state=50
)

## Build ML model without protected attributes as model features

# Apply one hot encoding to categorical columns (auto-detect object columns) and random forest model in the pipeline
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.87, random_state=50
)
output = x_test.copy()  # x_test df with output columns, to be appended later
x_train = x_train.drop(["age", "gender"], axis=1)
x_test = x_test.drop(["age", "gender"], axis=1)

estimator = Pipeline(
    steps=[
        ("onehot", ce.OneHotEncoder(use_cat_names=True)),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=3, max_features="sqrt", random_state=880
            ),
        ),
    ]
)

# Fit, predict and compute performance metrics
estimator.fit(x_train, y_train)

y_pred = estimator.predict(x_test)
y_probas = estimator.predict_proba(x_test)[::, 1]

precision_train = round(precision_score(y_train, estimator.predict(x_train)), 3)
recall_train = round(recall_score(y_train, estimator.predict(x_train)), 3)
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
ConfusionMatrixDisplay.from_estimator(estimator, x_train, y_train)
confusion_matrix_train = plot_to_str()
RocCurveDisplay.from_estimator(estimator, x_train, y_train)
roc_curve_train = plot_to_str()

# Test set
ConfusionMatrixDisplay.from_estimator(estimator, x_test, y_test)
confusion_matrix_test = plot_to_str()
RocCurveDisplay.from_estimator(estimator, x_test, y_test)
roc_curve_test = plot_to_str()


## Run some FEAT Tests on the data

# ROC/Min Max Threshold Test
smt_test = MinMaxMetricThreshold(
    # test_name="",        # Default test name and description will be used accordingly if not specified
    # test_desc="",
    attr="gender",
    metric="fpr",
    threshold=0.025,
    # proba_threshold = 0.6  # Custom probability threshold, default at 0.5
)
smt_test.run(df_test_with_output=output)
smt_test.plot()

smt_test2 = MinMaxMetricThreshold(
    attr="age",
    metric="fpr",
    threshold=0.025,
)
smt_test2.run(df_test_with_output=output)
smt_test2.plot()


# Subgroup Disparity Test
sgd_test = SubgroupDisparity(
    attr="age",
    metric="fpr",
    method="ratio",
    threshold=1.5,
)
sgd_test.run(output)
sgd_test.plot(alpha=0.05)  # default alpha argument shows 95% C.I bands

sgd_test2 = SubgroupDisparity(
    attr="gender",
    metric="fpr",
    method="ratio",
    threshold=1.5,
)
sgd_test2.run(output)
sgd_test2.plot(alpha=0.05)  # default alpha argument shows 95% C.I bands


# User inputted Feature importance test
imp_test = FeatureImportance(attrs=["gender", "age"], threshold=10)
imp_test.run(df_importance)
imp_test.plot(df_importance, show_n=10)  # Show top 10 most important features

# Initialize the mct and scaffold using the existing protobuf, for model 1
mct2 = mctlib.ModelCardToolkit(
    output_dir="model_card_output", file_name="credit_card_fraud_example2"
)
mc2 = mct2.scaffold_assets(path="initial_model_card.proto")
mc2.model_details.name = (
    "Credit Card Fraud Model, without protected attributes as model features"
)


## Convert test objects to a model-card-compatible format

# init model card test objects
mc_smt_test = mctlib.Test()
mc_smt_test2 = mctlib.Test()
mc_sgd_test = mctlib.Test()
mc_sgd_test2 = mctlib.Test()
mc_imp_test = mctlib.Test()

# assign tests to them
mc_smt_test.read_model_test(smt_test)
mc_smt_test2.read_model_test(smt_test2)
mc_sgd_test.read_model_test(sgd_test)
mc_sgd_test2.read_model_test(sgd_test2)
mc_imp_test.read_model_test(imp_test)

# Create 4 PerformanceMetric to store our results
mc2.quantitative_analysis.performance_metrics = [
    mctlib.PerformanceMetric() for i in range(0, 4)
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

# Add other components of a test (e.g. on explainability, fairness test) in a report
mc2.explainability_analysis.explainability_reports = [
    mctlib.ExplainabilityReport(
        type="Top 10 most important features",
        graphics=mctlib.GraphicsCollection(
            collection=[
                mctlib.Graphic(name=n, image=i) for n, i in imp_test.plots.items()
            ]
        ),
    )
]

mc2.fairness_analysis.fairness_reports[0].tests = [mc_smt_test, mc_smt_test2]
mc2.fairness_analysis.fairness_reports[1].tests = [mc_sgd_test, mc_sgd_test2]


mct2.update_model_card(mc2)
