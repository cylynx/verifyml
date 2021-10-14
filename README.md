# VerifyML

VerifyML is an opinionated, open-source toolkit and workflow to help companies implement human-centric AI practices. It is built on 3 principles:

- A git and code first approach to model development and maintenance.
- Automatic generation of model cards - machine learning documents that provide context and transparency into a model's development and performance.
- Model tests for validating performance of models across protected groups of interest, during development and in production.

## Components

![](https://github.com/cylynx/verifyml/blob/main/verifyml-dataflow.png)

At the core of the VerifyML workflow is a model card that captures 6 aspects of a model:

- Model details
- Considerations
- Model / data parameters
- Quantitative analysis
- Explainability analysis
- Fairness analysis

It is adapted from Google's Model Card Toolkit and expanded to include broader considerations such as fairness and explainability.

A web form - see [example form](https://tally.so/r/mR4Nlw), helps gather input and align stakeholders across product, data science, compliance.

Our Python toolkit supports data science workflows, and allows a custom model to be built and logged within the model card framework. The package also contains perfomance and fairness tests for model diagnostics, fairness and reliability checks.

Being a standard protobuf format, the model card can be translated to various outputs including a model report, trade-off comparison and even tests results summary.

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

### Save and export to html

```py
html = mct.export_format(output_file="example.html")
display.display(display.HTML(html))
```

## Model Tests

Model tests provides an out of the box way to conduct checks and analysis on performance, explainability and fairness. The tests included in VerifyML are atomic functions that can be imported and run without a model card. However, by using it with a model card, it provides a way to standardize objectives and check for intended or unintended model biases. It also automates documentation and renders the insights to a business friendly report.

Currently, VerifyML provides 5 classes of tests:

1. **Subgroup Disparity Test** - For a given metric, assert that the difference between the best and worst performing group is less than a specified threshold
2. **Min/Max Metric Threshold Test** - For a given metric, assert that all groups should be below / above a specified threshold
3. **Perturbation Test** - Assert that a given metric does not change significantly after perturbing on a specified input variable
4. **Feature Importance Test** - Assert that certain specified variables are not included as the top n most important features
5. **Data Shift Test** - Assert that the distributions of specified attributes are similar across two given datasets of interest

The detailed [model tests readme](https://github.com/cylynx/verifyml/blob/main/verifyml/model_tests/README.md) contains more information on the tests.

You can also easily create your own model tests by inheriting from the base model test class. See [DEVELOPMENT](https://github.com/cylynx/verifyml/blob/main/DEVELOPMENT.md) for more details.

### Example usage

```py
from verifyml.model_tests.FEAT import SubgroupDisparity

# Ratio of false positive rates between age subgroups should not be more than 1.5
sgd_test = SubgroupDisparity(metric='fpr', method='ratio', threshold=1.5)
sgd_test.run(output) # test data with prediction results
sgd_test.plot(alpha=0.05)
```

### Adding the test to the model card

```py
import verifyml.model_card_toolkit as mctlib

mc_sgd_test = mctlib.Test()
mc_sgd_test.read_model_test(sgd_test)
model_card.fairness_analysis.fairness_reports[0].tests = [mc_smt_test]
```

## Schema

Model cards are stored as a protobuf format. The reference model card protobuf schema can be found in the [proto directory](https://github.com/cylynx/verifyml/tree/main/verifyml/model_card_toolkit/proto). A translated copy in json schema format is also made available for convenience in the [schema folder](https://github.com/cylynx/verifyml/tree/main/verifyml/model_card_toolkit/schema)

## Templates

Model cards can be rendered into various reports through the use of templates. The template folder contains two html templates - a default model report and a compare template, and a default markdown model report.

## Contributions and Development

Contributions are always welcome - check out [CONTRIBUTING](https://github.com/cylynx/verifyml/blob/main/CONTRIBUTING.md)

The package and it's functionalities can be easily extended to meet the needs of a team. Check out [DEVELOPMENT](https://github.com/cylynx/verifyml/blob/main/DEVELOPMENT.md) for more info.

## Prior Art

The model card in VerifyML is adapted from Google's [Model Card Toolkit](https://github.com/tensorflow/model-card-toolkit). It is backward compatible with v0.0.2 and expands on it by adding sections on explainability and fairness. You can specify the desired rendering template by specifying the `template_path` argument when calling the `mct.export_format` function. For example:

```py
mct.export_format(output_file="example.md", template_path="path_to_my_template")
```

View the [templates' README](https://github.com/cylynx/verifyml/blob/main/verifyml/model_card_toolkit/template/README.md) for more information on creating your own jinja templates.

## References

[1] https://arxiv.org/abs/1810.03993

## License

VerifyML is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/cylynx/verifyml/blob/main/LICENSE) for the full license text.
