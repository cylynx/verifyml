# Automated Checks with Github Actions

Automate model checks with the [generate verifyml report action](https://github.com/marketplace/actions/generate-verifyml-report). Use this to ensure that the updated model passes all required test cases before being committed to the main / master branch.

The Github action automatically reads the Protobuf dataset from a specified path in the local repository and displays the test results as a comment in the pull request. It also generates a [Model Card Viewer](https://report.verifyml.com/) link for users to view the model card data.

To use the action, simply create a `verifyml-reports.yml` file within `.github/workflows` with the following content:

```yaml
name: Generate VerifyML Report

on:
  pull_request:
    branches:
      - "*"

jobs:
  report-generation:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout to Current Branch
        uses: actions/checkout@v2

      - name: Generate VerifyML Report
        uses: cylynx/verifyml-reports@v1
        with:
          data-path: "path-to-model-card-file.proto"
```

Replace the data-path attribute with the path to your model card file.

Each pull request made will check all tests listed in the quantitative analysis, explainability analysis, or fairness analysis section of the model card and produce a summary report of the test results:

![](https://github.com/cylynx/verifyml/raw/main/github-actions.png)
