# VerifyML

VerifyML is an opinionated, open-source toolkit and workflow to help companies implement human-centric AI practices. It is built on 3 principles:

- A git and code first approach to model development and maintenance.
- Automatic generation of model cards - machine learning documents that provide context and transparency into a model's development and performance.
- Model tests for validating performance of models across protected groups of interests, during developing and in production.

## Workflow

![](verifyml-workflow.png)

The VerifyML workflow starts from model conceptualization, to model building and deployment. In a typical data science workflow, a model is typically developed by a data scientist with inputs from the business team. This means that typically a model serves to maximise the business objective without considerations of side effects, differential benefits and harms across groups, and performance degradation over time.

The VerifyML workflow introduces these concepts and trade-offs as part of the model lifecycle. By bringing these questions to the fore, teams gain the following benefits:

- Better clarity on of a model's outcome, potential side-effects, and areas of uncertanity
- Faster alignment across model builders, product owners and internal auditors
- Oversight and accountability

These qualitative inputs then get translated to code (where possible) and act as modelling constraints or considerations. The model card captures and logs artifacts relevant in the model development phase and allow such information to be easy shared across the organization.  

Tests relating to performance or fairness can also be included to ensure that the model meets the desired objective. This can be added to a CI/CD process where such tests are run on a regular basis to ensure that there is no unexpected drift in performance.

## Installation

The Model Card Toolkit is hosted on [PyPI](https://pypi.org/project/verifyml/), and can be installed with `pip install verifyml`.

## Getting Started
### Generate a model card

You can bootstrap a model card with our [tally web form](https://tally.so/r/mR4Nlw) or generate it with the python toolkit:

```py
import verifyml.model_card_toolkit as mctlib

# Initialize the Model Card Toolkit with a path to store generate assets
mct = mctlib.ModelCardToolkit(output_dir="model_card_output", file_name="breast_cancer_diagnostic_model_card")
model_card = mct.scaffold_assets()
```

### Populate the model card with details

```py
# You can add free text fields
model_card.model_details.name = 'Breast Cancer Wisconsin (Diagnostic) Dataset'

# Or use helper classes
model_card.model_parameters.data.append(mctlib.Dataset())
model_card.model_parameters.data[0].graphics.description = (
  f'{len(X_train)} rows with {len(X_train.columns)} features')
model_card.model_parameters.data[0].graphics.collection = [
    mctlib.Graphic(image=mean_radius_train),
    mctlib.Graphic(image=mean_texture_train)
]
```

### Add test cases

```py
from verifyml.model_tests.FEAT import SubgroupDisparity

# Ratio of false positive rates between age subgroups should not be more than 1.5
sgd_test = SubgroupDisparity(metric='fpr', method='ratio', threshold=1.5)
sgd_test.run(output) # test data with prediction results
sgd_test.plot(alpha=0.05)
```

### Save and export to html

```py
html = mct.export_format(output_file="example.html")
display.display(display.HTML(html))
```

## FEAT Tests

Add section on the variety of FEAT tests

## Schema

Model cards are stored as a protobuf format. You can see the model card protobuf schema in the [proto directory](verifyml/model_card_toolkit/proto). A translated copy in json schema format is also made available for convenience in the [schema folder](verifyml/model_card_toolkit/schema)

## Development

### Publishing to PyPI

- Set version number and configs in `setup.cfg`

```bash
pip install --upgrade setuptools build twine

# build package files
python -m build

# upload to testpypi
python -m twine upload --repository testpypi dist/*
```

## Prior Art

The model card in VeriyML is adpated from Google's [Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit). It is backward compatible with v0.0.2 and expands on it by adding sections on explainability and fairness.  

## References

[1] https://arxiv.org/abs/1810.03993

