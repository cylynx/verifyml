# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model Card Toolkit.

The Model Card Toolkit (MCT) provides a set of utilities to generate Model Cards
from trained models, evaluations, and datasets in ML pipelines.
"""

import os
import pkgutil
import tempfile
from typing import Optional, Text

from absl import logging
import jinja2
from google.protobuf import message

from .model_card import ModelCard
from .proto import model_card_pb2

# Constants about provided UI templates.
_UI_TEMPLATES = (
    "template/html/default_template.html.jinja",
    "template/md/default_template.md.jinja",
    "template/md/results_template.md.jinja",
)
_DEFAULT_UI_TEMPLATE_FILE = os.path.join("html", "default_template.html.jinja")
_DEFAULT_UI_RESULT_TEMPLATE_FILE = os.path.join("md", "results_template.md.jinja")

# Constants about Model Cards Toolkit Assets (MCTA).
_MCTA_PROTO_FILE = os.path.join("data", "model_card.proto")
_MCTA_TEMPLATE_DIR = "template"
_MCTA_RESOURCE_DIR = os.path.join("resources", "plots")
# Constants about the final generated model cards.
_MODEL_CARDS_DIR = "model_cards"
_DEFAULT_MODEL_CARD_FILE_NAME = "model_card.html"
_DEFAULT_MODEL_CARD_RESULTS_FILE_NAME = "model_card_results.md"


class ModelCardToolkit:
    """ModelCardToolkit provides utilities to generate a ModelCard.

    ModelCardToolkit is a tool for ML practitioners to create Model Cards,
    documentation for model information such as owners, use cases, training and
    evaluation data, performance, etc. A Model Card document can be displayed in
    output formats including HTML, Markdown, etc.

    The ModelCardToolkit includes an API designed for a human-in-the-loop process
    to elaborate the ModelCard.

    The ModelCardToolkit organizes the ModelCard assets (e.g., structured data,
    plots, and UI templates) in a user-specified directory, and updates them
    incrementally via its API.


    Example usage:

    ```python
    import model_card_toolkit

    # Initialize the Model Card Toolkit with a path to store generate assets
    model_card_output_path = ...
    mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)

    # Initialize the ModelCard, which can be freely populated
    model_card = mct.scaffold_assets()
    model_card.model_details.name = 'My Model'

    # Write the model card data to a proto file
    mct.update_model_card(model_card)

    # Return the model card document as an HTML page
    html = mct.export_format()
    ```
    """

    def __init__(
        self, output_dir: Optional[Text] = None, file_name: Optional[Text] = None
    ):
        """Initializes the ModelCardToolkit.

        This function does not generate any assets by itself. Use the other API
        functions to generate Model Card assets. See class-level documentation for
        example usage.

        Args:
          output_dir: The path where MCT assets (such as data files and model cards)
            are written to. If not provided, a temp directory is used.
          file_name: file name of the model card proto file. Defaults to model_card.proto
        """
        self.output_dir = output_dir or tempfile.mkdtemp()
        self.proto_file_name = (
            os.path.join(
                "data",
                file_name if file_name.endswith(".proto") else f"{file_name}.proto",
            )
            if file_name
            else _MCTA_PROTO_FILE
        )
        self._mcta_proto_file = os.path.join(self.output_dir, self.proto_file_name)
        self._mcta_template_dir = os.path.join(self.output_dir, _MCTA_TEMPLATE_DIR)
        self._model_cards_dir = os.path.join(self.output_dir, _MODEL_CARDS_DIR)

    def _jinja_loader(self, template_dir: Text):
        return jinja2.FileSystemLoader(template_dir)

    def _write_file(self, path: Text, content: Text) -> None:
        """Write content to the path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w+") as f:
            f.write(content)

    def _write_proto_file(self, path: Text, model_card: ModelCard) -> None:
        """Write serialized model card proto to the path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(model_card.to_proto().SerializeToString())

    def _read_proto_file(self, path: Text) -> ModelCard:
        """Read serialized model card proto from the path."""
        model_card_proto = model_card_pb2.ModelCard()
        with open(path, "rb") as f:
            model_card_proto.ParseFromString(f.read())
        return ModelCard().copy_from_proto(model_card_proto)

    def scaffold_assets(
        self, path: Optional[Text] = None, proto: Optional[message.Message] = None
    ) -> ModelCard:
        """Generates the Model Card Tookit assets.

        If a path to an existing model card proto object is provided,
        it will be copied over as the base card instead of initializing a new one.

        Alternatively, if an existing proto object is provided, it will be copied over as the base card.

        Assets include the ModelCard proto file, Model Card document, and jinja
        template. These are written to the `output_dir` declared at
        initialization.

        Args:
          path: The path to model card proto.
          model_card: The ModelCard object.

        Returns:
          A ModelCard representing the given model.

        Raises:
          FileNotFoundError: on failure to copy the template files.
        """

        # If path exist, read proto from path
        if path and os.path.exists(path):
            model_card = self._read_proto_file(path)
        # If proto, bootstrap model card from proto
        elif proto:
            model_card_proto = model_card_pb2.ModelCard()
            model_card_proto.ParseFromString(proto)
            model_card = ModelCard().copy_from_proto(model_card_proto)
        else:
            model_card = ModelCard()

        # Write Proto file.
        self._write_proto_file(self._mcta_proto_file, model_card)

        # Write UI template files.
        for template_path in _UI_TEMPLATES:
            template_content = pkgutil.get_data("model_card_toolkit", template_path)
            if template_content is None:
                raise FileNotFoundError(f"Cannot find file: '{template_path}'")
            template_content = template_content.decode("utf8")
            self._write_file(
                os.path.join(self.output_dir, template_path), template_content
            )

        return model_card

    def update_model_card(self, model_card: ModelCard) -> None:
        """Updates the Proto file in the MCT assets directory.

        Args:
          model_card: The updated model card that users want to write back.

        Raises:
           Error: when the given model_card is invalid w.r.t. the schema.
        """
        self._write_proto_file(self._mcta_proto_file, model_card)

    def export_format(
        self,
        model_card: Optional[ModelCard] = None,
        template_path: Optional[Text] = None,
        output_file=_DEFAULT_MODEL_CARD_FILE_NAME,
    ) -> Text:
        """Generates a model card document based on the MCT assets.

        The model card document is both returned by this function, as well as saved
        to output_file.

        Args:
          model_card: The ModelCard object, generated from `scaffold_assets()`. If
            not provided, it will be read from the ModelCard proto file in the
            assets directory.
          template_path: The file path of the Jinja template. If not provided, the
            default template will be used.
          output_file: The file name of the generated model card. If not provided,
            the default 'model_card.html' will be used. If the file already exists,
            then it will be overwritten.

        Returns:
          The model card file content.

        Raises:
          MCTError: If `export_format` is called before `scaffold_assets` has
            generated model card assets.
        """
        if not template_path:
            template_path = os.path.join(
                self._mcta_template_dir, _DEFAULT_UI_TEMPLATE_FILE
            )
        template_dir = os.path.dirname(template_path)
        template_file = os.path.basename(template_path)

        # If model_card is passed in, write to Proto file.
        if model_card:
            self.update_model_card(model_card)
        # If model_card is not passed in, read from Proto file.
        elif os.path.exists(self._mcta_proto_file):
            model_card = self._read_proto_file(self._mcta_proto_file)
        # If model card proto never created, raise exception.
        else:
            raise ValueError("scaffold_assets() must be called before export_format().")

        # Generate Model Card.
        jinja_env = jinja2.Environment(
            loader=self._jinja_loader(template_dir),
            autoescape=True,
            auto_reload=True,
            cache_size=0,
        )
        template = jinja_env.get_template(template_file)
        model_card_file_content = template.render(
            model_details=model_card.model_details,
            model_parameters=model_card.model_parameters,
            quantitative_analysis=model_card.quantitative_analysis,
            explainability_analysis=model_card.explainability_analysis,
            fairness_analysis=model_card.fairness_analysis,
            considerations=model_card.considerations,
        )

        # Write the model card document file and return its contents.
        mode_card_file_path = os.path.join(self._model_cards_dir, output_file)
        self._write_file(mode_card_file_path, model_card_file_content)

        return model_card_file_content


    def export_test_results(
        self,
        model_card: ModelCard,
        template_path: Optional[Text] = None,
        output_file=_DEFAULT_MODEL_CARD_RESULTS_FILE_NAME,
    ) -> Text:
        """Generates a document containing model card test results.

        The model card result document is both returned by this function, as well as saved
        to output_file.

        Args:
          model_card: The ModelCard object, generated from `scaffold_assets()`. If
            not provided, it will be read from the ModelCard proto file in the
            assets directory.
          template_path: The file path of the Jinja template. If not provided, the
            default template will be used.
          output_file: The file name of the generated model card. If not provided,
            the default 'model_card.html' will be used. If the file already exists,
            then it will be overwritten.

        Returns:
          The model card file content.

        Raises:
          MCTError: If `export_format` is called before `scaffold_assets` has
            generated model card assets.
        """
        if not template_path:
            template_path = os.path.join(
                self._mcta_template_dir, _DEFAULT_UI_RESULT_TEMPLATE_FILE
            )
        template_dir = os.path.dirname(template_path)
        template_file = os.path.basename(template_path)

        # Generate Model Card.
        jinja_env = jinja2.Environment(
            loader=self._jinja_loader(template_dir),
            autoescape=True,
            auto_reload=True,
            cache_size=0,
        )
        template = jinja_env.get_template(template_file)
        model_card_file_content = template.render(
            test_results=model_card.get_test_results(),
        )

        # Write the model card document file and return its contents.
        mode_card_file_path = os.path.join(self._model_cards_dir, output_file)
        self._write_file(mode_card_file_path, model_card_file_content)

        return model_card_file_content
