# Examples

The examples listed here can either be run by cloning this repository or downloading the `verifyml` package:

```py
pip install verifyml
```

## Example Notebooks

1. `getting_started.ipynb` - A quick overview of the model card approach. Companion notebook to the [introductory post](https://medium.com/cylynx/a-quickstart-guide-to-verifyml-pt-1-c1a751194a68)

2. `sklearn_model_card_example.ipynb` - Gives a good overview of the functionalities of the toolkit. Demonstrates using the model card to document performance and explainability metrics and exporting a html report card.

3. `model_card_editor.ipynb` - Imports a protobuf generated from our [model card editor](https://report.verifyml.com) - `initial_model_card.proto`, and exporting a modified protobuf file to html.

4. `credit_card_fraud_example.ipynb` - A more detailed example of how the model card can be used to assess fairness objectives and compare performance across models. This notebook also introduces some of the fairness, ethics, accountability and transparency tests that is part of the `verifyml` package and integrates in the model card. The example is modelled after a hypothetical credit card fraud classification model based on mock transactional data.

5. `loan_approval_example.ipynb` - The example is modelled after a hypothetical loan approval classification model based on real transactional data.

6. `credit_risk_regression_example.ipynb` - The example is modelled after a hypothetical credit risk (regression problem) model based on real transactional data.
