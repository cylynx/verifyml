# DEVELOPMENT

Here are some guidelines on extending certain common functionalities of the package.

## Installing Locally

```sh
pip install . --force-reinstall
```

`--force-reinstall` is only needed if you've previously downloaded VerifyML with `pip`.

## Model Card Templates

We use [Jinja](https://jinja.palletsprojects.com/en/3.0.x/) templates to create the HTML needed to display Model Cards. For details on how the templates are used, refer to the [templates' README](verifyml/model_card_toolkit/template/README.md).

If you've created new templates and would like them to be included in the output directory when `scaffold_assets()` is called, update the `_UI_TEMPLATES` variable in [model_card_toolkit.py](verifyml/model_card_toolkit/model_card_toolkit.py):

```python
_UI_TEMPLATES = (
    "template/html/default_template.html.jinja",
    "template/html/compare.html.jinja",
    "template/md/default_template.md.jinja",
    ...,    # your new template name
)
```

## Model Tests

If the [provided model tests](verifyml/model_tests) are insufficient, you can easily create your own by extending the [ModelTest](verifyml/model_tests/ModelTest.py) abstract base class.

When creating a subclass from ModelTest, there are a few requirements for the test to render properly on a Model Card:

1. It must have a `run()` method. The method should do these things:
   - Run your test and save the results into `self.result`
   - Update `self.passed`, a boolean indicating if the test result passes your defined condition
2. If plots are to be displayed, the subclass also needs a method (e.g. `plot()`) that stores them in `self.plots` as base64-encoded strings
3. Optionally, the subclass can overwrite the base class' `test_name` and `test_desc` to display that information in the model card as well

Refer to the Model Test [DEVELOPMENT.ipynb](verifyml/model_tests/DEVELOPMENT.ipynb) notebook to see a toy example of how this can be done.

## Model Card Protobuf Schema

The protobuf schema is located [here](verifyml/model_card_toolkit/proto/model_card.proto), and is used to convert model card data saved in protobuf format to and from a Python-usable format. If there is a need to modify it, it is recommended to add new protobuf fields instead of overwriting existing ones, to preserve backward compatibility.

To use it, you need to first install [`bazel`](https://docs.bazel.build/versions/4.2.1/install.html).

After that, run this from the `verifyml` folder:

```sh
bazel run //model_card_toolkit:move_generated_files
```

This will read the protobuf schema and create a [`model_card_pb2.py` file](verifyml/model_card_toolkit/proto/model_card_pb2.py) that can be imported and used in Python.

For example, this is what the schema looks like for defining model owners:

```protobuf
// model_card.proto

// The information about owners of a model.
// The next tag number is 4.
message Owner {
  // The name of the model owner.
  optional string name = 1;

  // The contact information for the model owner or owners.
  optional string contact = 2;

  // The role of the person e.g. developer, owner, auditor.
  optional string role = 3;
}
```

Running the `bazel` command will create a `model_card_pb2.py` file that contains this information. A [corresponding Python class](verifyml/model_card_toolkit/model_card.py) can use the definition like this:

```python
import dataclasses

from .base_model_card_field import BaseModelCardField
from .proto import model_card_pb2

@dataclasses.dataclass
class Owner(BaseModelCardField):
    name: Optional[str] = None
    contact: Optional[str] = None
    role: Optional[str] = None

    _proto_type: dataclasses.InitVar[type(model_card_pb2.Owner)] = model_card_pb2.Owner
```

### Protobuf to JSON Schema

The protobuf schema can be converted into a JSON schema with this command:

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
