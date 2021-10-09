# DEVELOPMENT

Here are some guidelines on extending certain common functionalities of the package

## Templates

How to create new template - recommend user to just create a new jinja template and point to it. Also mention template dir.

## Model Tests

Introduction to the base `ModelTest` class and how to extend it to create a custom test case.

## Protobuf Schema

The reference protobuf schema is located [here](verifyml/model_card_toolkit/proto/model_card.proto). Recommend not to remove fields but add new ones of interest, if really required.

To build, run from `verifyml` folder:

```sh
bazel run //model_card_toolkit:move_generated_files
```

The associated json schema can subsequently be generated with:

```sh
bash model_card_toolkit/proto/gen_json_schema.sh
```

## Publishing to PyPI

- Set version number and configs in `setup.cfg`

```bash
pip install --upgrade setuptools build twine

# build package files
python -m build

# upload to pypi
python -m twine upload --repository [pypi/testpypi] dist/*
```
