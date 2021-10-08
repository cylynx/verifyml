# Examples

The examples listed here can either be run by cloning this repository or downloading the `verifyml` package:

```py
pip install verifyml
```

## Example Notebooks

1) `sklearn_model_card_example.ipynb` - Gives a good overview of the functionalities of the toolkit. Demonstrates using the model card to document performance and explainability metrics and exporting a html report card.

2) `tally_form_example.ipynb` - Takes a response generated from our [tally web form](https://tally.so/r/mR4Nlw) and converts it to a model card in 3 lines of code. It uses `sample-form-response.json` which is a sample response from the web form.

3) `credit_card_fraud_example.ipynb` - A more detailed example of how the model card can be used to assess fairness objectives and compare performance across models. This notebook also introduces some of the fairness, ethics, accountability and transparency tests that is part of the `verifyml` package and integrates in the model card. The example is modelled after a hypothetical credit card fraud classification model based on mock transactional data.
