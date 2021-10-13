# Copyright 2021 Cylynx
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
"""Model Card Data Class.

The Model Card (MC) is the document designed for transparent reporting of AI
model provenance, usage, and ethics-informed evaluation. The model card can be
presented by different formats (e.g. HTML, PDF, Markdown). The properties of
the Model Card (MC) are defined by a json schema. The ModelCard class in the
ModelCardsToolkit serves as an API to read and write MC properties by the users.
"""

import dataclasses
import pandas as pd
import json as json_lib
import collections
from typing import Any, Dict, List, Optional, Counter

from .base_model_card_field import BaseModelCardField
from .proto import model_card_pb2
from .utils import validation

_SCHEMA_VERSION_STRING = "schema_version"


@dataclasses.dataclass
class Owner(BaseModelCardField):
    """The information about owners of a model.

    Attributes:
      name: The name of the model owner.
      contact: The contact information for the model owner or owners. These could
        be individual email addresses, a team mailing list expressly, or a
        monitored feedback form.
      role: The role of the person e.g. owner, developer or auditor.
    """

    name: Optional[str] = None
    contact: Optional[str] = None
    role: Optional[str] = None

    _proto_type: dataclasses.InitVar[type(model_card_pb2.Owner)] = model_card_pb2.Owner


@dataclasses.dataclass
class Version(BaseModelCardField):
    """The information about verions of a model.

    If there are multiple versions of the model, or there may be in the future,
    it’s useful for your audience to know which version of the model is
    discussed
    in the Model Card. If there are previous versions of this model, briefly
    describe how this version is different. If no more than one version of the
    model will be released, this field may be omitted.

    Attributes:
      name: The name of the version.
      date: The date this version was released.
      diff: The changes from the previous version.
    """

    name: Optional[str] = None
    date: Optional[str] = None
    diff: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.Version)
    ] = model_card_pb2.Version


@dataclasses.dataclass
class License(BaseModelCardField):
    """The license information for a model.

    Attributes:
      identifier: A standard SPDX license identifier (https://spdx.org/licenses/),
        or "proprietary" for an unlicensed Module.
      custom_text: The text of a custom license.
    """

    identifier: Optional[str] = None
    custom_text: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.License)
    ] = model_card_pb2.License


@dataclasses.dataclass
class Reference(BaseModelCardField):
    """Reference for a model.

    Attributes:
      reference: A reference to a resource.
    """

    reference: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.Reference)
    ] = model_card_pb2.Reference


@dataclasses.dataclass
class Citation(BaseModelCardField):
    """A citation for a model.

    Attributes:
      style: The citation style, such as MLA, APA, Chicago, or IEEE.
      citation: the citation.
    """

    style: Optional[str] = None
    citation: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.Citation)
    ] = model_card_pb2.Citation


@dataclasses.dataclass
class ModelDetails(BaseModelCardField):
    """This section provides a general, high-level description of the model.

    Attributes:
      name: The name of the model.
      overview: A description of the model card.
      documentation: A more thorough description of the model and its usage.
      owners: The individuals or teams who own the model.
      version: The version of the model.
      licenses: The license information for the model. If the model is licensed
        for use by others, include the license type. If the model is not licensed
        for future use, you may state that here as well.
      references: Provide any additional links the reader may need. You can link
        to foundational research, technical documentation, or other materials that
        may be useful to your audience.
      citations: How should the model be cited? If the model is based on published
        academic research, cite the research.
      regulatory_requirements: Provide any regulatory requirements that the model should comply to.
    """

    name: Optional[str] = None
    overview: Optional[str] = None
    documentation: Optional[str] = None
    owners: List[Owner] = dataclasses.field(default_factory=list)
    version: Optional[Version] = dataclasses.field(default_factory=Version)
    licenses: List[License] = dataclasses.field(default_factory=list)
    references: List[Reference] = dataclasses.field(default_factory=list)
    citations: List[Citation] = dataclasses.field(default_factory=list)
    regulatory_requirements: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.ModelDetails)
    ] = model_card_pb2.ModelDetails


@dataclasses.dataclass
class Graphic(BaseModelCardField):
    """A named inline plot.

    Attributes:
      name: The name of the graphic.
      image: The image string encoded as a base64 string.
    """

    name: Optional[str] = None
    image: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.Graphic)
    ] = model_card_pb2.Graphic


@dataclasses.dataclass
class GraphicsCollection(BaseModelCardField):
    """A collection of graphics.

    Each ```graphic``` in the ```collection``` field has both a ```name``` and
    an ```image```. For instance, you might want to display a graph showing the
    number of examples belonging to each class in your training dataset:

    ```python

    model_card.model_parameters.data.train.graphics.collection = [
      {'name': 'Training Set Size', 'image': training_set_size_barchart},
    ]
    ```

    Then, provide a description of the graph:

    ```python

    model_card.model_parameters.data.train.graphics.description = (
      'This graph displays the number of examples belonging to each class ',
      'in the training dataset. ')
    ```

    Attributes:
      description: The description of graphics.
      collection: A collection of graphics.
    """

    description: Optional[str] = None
    collection: List[Graphic] = dataclasses.field(default_factory=list)

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.GraphicsCollection)
    ] = model_card_pb2.GraphicsCollection


@dataclasses.dataclass
class SensitiveData(BaseModelCardField):
    """Sensitive data, such as PII (personally-identifiable information).

    Attributes:
      sensitive_data: A description of any sensitive data that may be present in a
        dataset. Be sure to note PII information such as names, addresses, phone
        numbers, etc. Preferably, such info should be scrubbed from a dataset if
        possible. Note that even non-identifying information, such as zip code,
        age, race, and gender, can be used to identify individuals when
        aggregated. Please describe any such fields here.
      sensitive_data_used: A list of sensitive data used in the deployed model.
      justification: Justification of the need to use the fields in deployment.
    """

    sensitive_data: List[str] = dataclasses.field(default_factory=list)
    sensitive_data_used: List[str] = dataclasses.field(default_factory=list)
    justification: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.SensitiveData)
    ] = model_card_pb2.SensitiveData


@dataclasses.dataclass
class Dataset(BaseModelCardField):
    """Provide some information about a dataset used to generate a model.

    Attributes:
      name: The name of the dataset.
      description: The description of dataset.
      link: A link to the dataset.
      sensitive: Does this dataset contain human or other sensitive data?
      graphics: Visualizations of the dataset.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    link: Optional[str] = None
    sensitive: Optional[SensitiveData] = dataclasses.field(
        default_factory=SensitiveData
    )
    graphics: GraphicsCollection = dataclasses.field(default_factory=GraphicsCollection)

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.Dataset)
    ] = model_card_pb2.Dataset


@dataclasses.dataclass
class ModelParameters(BaseModelCardField):
    """Parameters for construction of the model.

    Attributes:
      model_architecture: specifies the architecture of your model.
      data: specifies the datasets used to train and evaluate your model.
      input_format: describes the data format for inputs to your model.
      output_format: describes the data format for outputs from your model.
    """

    model_architecture: Optional[str] = None
    data: List[Dataset] = dataclasses.field(default_factory=list)
    input_format: Optional[str] = None
    output_format: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.ModelParameters)
    ] = model_card_pb2.ModelParameters


@dataclasses.dataclass
class Test(BaseModelCardField):
    """Information about test that is runned against the model.

    Attributes:
      name: The name of the test.
      description: User-friendly description of the test.
      threshold: Threshold required to pass the test.
      result: Result returned by the test.
      passed: Whether the model result satisfies the given threshold.
      graphics: A collection of visualizations associated with the test.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    threshold: Optional[str] = None
    result: Optional[str] = None
    passed: Optional[bool] = None
    graphics: GraphicsCollection = dataclasses.field(default_factory=GraphicsCollection)

    _proto_type: dataclasses.InitVar[type(model_card_pb2.Test)] = model_card_pb2.Test

    def read_model_test(self, model_test) -> None:
        self.name = model_test.test_name
        self.description = model_test.test_desc
        self.threshold = str(getattr(model_test, "threshold", None))
        self.result = (
            model_test.result.to_csv(index=True)
            if isinstance(model_test.result, pd.DataFrame)
            else str(model_test.result)
        )
        self.passed = model_test.passed

        plots = getattr(model_test, "plots", None)
        if plots:
            collection = [Graphic(name=n, image=i) for n, i in plots.items()]
            self.graphics = GraphicsCollection(collection=collection)


@dataclasses.dataclass
class PerformanceMetric(BaseModelCardField):
    """The details of the performance metric.

    Attributes:
      type: What performance metric are you reporting on?
      value: What is the value of this performance metric?
      slice: What slice of your data was this metric computed on?
      description: User-friendly description of the performance metric.
      graphics: A collection of visualizations associated with the metric.
      tests: A collection of tests associated with the metric.
    """

    type: Optional[str] = None
    value: Optional[str] = None
    slice: Optional[str] = None
    description: Optional[str] = None
    graphics: GraphicsCollection = dataclasses.field(default_factory=GraphicsCollection)
    tests: List[Test] = dataclasses.field(default_factory=list)

    _proto_type: dataclasses.InitVar[
        BaseModelCardField._get_type(model_card_pb2.PerformanceMetric)
    ] = model_card_pb2.PerformanceMetric


@dataclasses.dataclass
class QuantitativeAnalysis(BaseModelCardField):
    """The quantitative analysis of a model.

    Identify relevant performance metrics and display values. Let’s say you’re
    interested in displaying the accuracy and false positive rate (FPR) of a
    cat vs. dog classification model. Assuming you have already computed both
    metrics, both overall and per-class, you can specify metrics like so:

    ```python
    model_card.quantitative_analysis.performance_metrics = [
      {'type': 'accuracy', 'value': computed_accuracy},
      {'type': 'accuracy', 'value': cat_accuracy, 'slice': 'cat'},
      {'type': 'accuracy', 'value': dog_accuracy, 'slice': 'dog'},
      {'type': 'fpr', 'value': computed_fpr},
      {'type': 'fpr', 'value': cat_fpr, 'slice': 'cat'},
      {'type': 'fpr', 'value': dog_fpr, 'slice': 'dog'},
    ]
    ```

    Attributes:
      performance_metrics: The performance metrics being reported.
      graphics: A collection of visualizations of model performance.
    """

    performance_metrics: List[PerformanceMetric] = dataclasses.field(
        default_factory=list
    )
    graphics: GraphicsCollection = dataclasses.field(default_factory=GraphicsCollection)

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.QuantitativeAnalysis)
    ] = model_card_pb2.QuantitativeAnalysis


@dataclasses.dataclass
class ExplainabilityReport(BaseModelCardField):
    """Model explainability report.

    Details of how the model works such as feature importance,
    decision trees or LIME or shapely analysis.

    Attributes:
      type: What explainability method are you conducting?
      slice: What slice of your data was this analysis conducted on?
      description: User-friendly description of the explainability metric.
      graphics: A collection of visualizations related to the explainability method.
      tests: A collection of tests associated with the explainability method.
    """

    type: Optional[str] = None
    slice: Optional[str] = None
    description: Optional[str] = None
    graphics: GraphicsCollection = dataclasses.field(default_factory=GraphicsCollection)
    tests: List[Test] = dataclasses.field(default_factory=list)

    _proto_type: dataclasses.InitVar[
        BaseModelCardField._get_type(model_card_pb2.ExplainabilityReport)
    ] = model_card_pb2.ExplainabilityReport


@dataclasses.dataclass
class ExplainabilityAnalysis(BaseModelCardField):
    """Model explainability.

    Analysis to explain how the model works and operates.

    Attributes:
      explainability_reports: The explainability studies undertaken.
    """

    explainability_reports: List[ExplainabilityReport] = dataclasses.field(
        default_factory=list
    )

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.ExplainabilityAnalysis)
    ] = model_card_pb2.ExplainabilityAnalysis


@dataclasses.dataclass
class FairnessReport(BaseModelCardField):
    """Model fairness report.

    Details on fairness checks and analysis.

    Attributes:
      type: What fairness assessment method are you conducting?
      slice: What slice of your data was this analysis conducted on?
      segment: What segment of the dataset which the fairness report is assessing?
      description: User-friendly description of the fairness metric.
      graphics: A collection of visualizations related to the fairness method.
      tests: A collection of tests associated with the fairness method.
    """

    type: Optional[str] = None
    slice: Optional[str] = None
    segment: Optional[str] = None
    description: Optional[str] = None
    graphics: GraphicsCollection = dataclasses.field(default_factory=GraphicsCollection)
    tests: List[Test] = dataclasses.field(default_factory=list)

    _proto_type: dataclasses.InitVar[
        BaseModelCardField._get_type(model_card_pb2.FairnessReport)
    ] = model_card_pb2.FairnessReport


@dataclasses.dataclass
class FairnessAnalysis(BaseModelCardField):
    """Model fairness.

    Attributes:
      fairness_reports: The fairness studies undertaken.
    """

    fairness_reports: List[FairnessReport] = dataclasses.field(default_factory=list)

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.FairnessAnalysis)
    ] = model_card_pb2.FairnessAnalysis


@dataclasses.dataclass
class User(BaseModelCardField):
    """A type of user for a model.

    Attributes:
      description: A description of a user.
    """

    description: Optional[str] = None

    _proto_type: dataclasses.InitVar[type(model_card_pb2.User)] = model_card_pb2.User


@dataclasses.dataclass
class UseCase(BaseModelCardField):
    """A type of use case for a model.

    Attributes:
      description: A description of a use case.
    """

    description: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.UseCase)
    ] = model_card_pb2.UseCase


@dataclasses.dataclass
class Limitation(BaseModelCardField):
    """A limitation a model.

    Attributes:
      description: A description of the limitation.
    """

    description: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.Limitation)
    ] = model_card_pb2.Limitation


@dataclasses.dataclass
class Tradeoff(BaseModelCardField):
    """A tradeoff for a model.

    Attributes:
      description: A description of the tradeoff.
    """

    description: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.Tradeoff)
    ] = model_card_pb2.Tradeoff


@dataclasses.dataclass
class Risk(BaseModelCardField):
    """Information about risks involved when using the model.

    Attributes:
      name: The name of the risk.
      mitigation_strategy: A mitigation strategy that you've implemented, or one
        that you suggest to users.
    """

    name: Optional[str] = None
    mitigation_strategy: Optional[str] = None

    _proto_type: dataclasses.InitVar[type(model_card_pb2.Risk)] = model_card_pb2.Risk


@dataclasses.dataclass
class FairnessAssessment(BaseModelCardField):
    """Information about the benefits and harms of the model.

    Attributes:
      group_at_risk: The groups or individuals at risk of being systematically
        disadvantaged by the model.
      benefits: Expected benefits to the identified groups.
      harms: Expected harms to the identified groups.
      mitigation_strategy: With respect to the benefits and harms outlined,
        please describe any mitigation strategy implemented.
    """

    group_at_risk: Optional[str] = None
    benefits: Optional[str] = None
    harms: Optional[str] = None
    mitigation_strategy: Optional[str] = None

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.FairnessAssessment)
    ] = model_card_pb2.FairnessAssessment


@dataclasses.dataclass
class Considerations(BaseModelCardField):
    """Considerations related to model construction, training, and application.

    The considerations section includes qualitative information about your model,
    including some analysis of its risks and limitations. As such, this section
    usually requires careful consideration, and conversations with many relevant
    stakeholders, including other model developers, dataset producers, and
    downstream users likely to interact with your model, or be affected by its
    outputs.

    Attributes:
      users: Who are the intended users of the model? This may include
        researchers, developers, and/or clients. You might also include
        information about the downstream users you expect to interact with your
        model.
      use_cases: What are the intended use cases of the model? What use cases are
        out-of-scope?
      limitations: What are the known limitations of the model? This may include
        technical limitations, or conditions that may degrade model performance.
      tradeoffs: What are the known accuracy/performance tradeoffs for the model?
      ethical_considerations: What are the ethical risks involved in application
        of this model? For each risk, you may also provide a mitigation strategy
        that you've implemented, or one that you suggest to users.
      fairness_assessment: How does the model affect groups at risk of being
        systematically disadvantaged? What are the harms and benefits to the various
        affected groups?
    """

    users: List[User] = dataclasses.field(default_factory=list)
    use_cases: List[UseCase] = dataclasses.field(default_factory=list)
    limitations: List[Limitation] = dataclasses.field(default_factory=list)
    tradeoffs: List[Tradeoff] = dataclasses.field(default_factory=list)
    ethical_considerations: List[Risk] = dataclasses.field(default_factory=list)
    fairness_assessment: List[FairnessAssessment] = dataclasses.field(
        default_factory=list
    )

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.Considerations)
    ] = model_card_pb2.Considerations


@dataclasses.dataclass
class ModelCard(BaseModelCardField):
    """Fields used to generate the Model Card.

    Attributes:
      model_details: Descriptive metadata for the model.
      model_parameters: Technical metadata for the model.
      quantitative_analysis: Quantitative analysis of model performance.
      considerations: Any considerations related to model construction, training,
        and application.
      explainability_analysis: Explainability analysis being reported.
      fairness_analysis: Fairness analysis being reported.
    """

    model_details: ModelDetails = dataclasses.field(default_factory=ModelDetails)
    model_parameters: ModelParameters = dataclasses.field(
        default_factory=ModelParameters
    )
    quantitative_analysis: QuantitativeAnalysis = dataclasses.field(
        default_factory=QuantitativeAnalysis
    )
    considerations: Considerations = dataclasses.field(default_factory=Considerations)
    explainability_analysis: ExplainabilityAnalysis = dataclasses.field(
        default_factory=ExplainabilityAnalysis
    )
    fairness_analysis: FairnessAnalysis = dataclasses.field(
        default_factory=FairnessAnalysis
    )

    _proto_type: dataclasses.InitVar[
        type(model_card_pb2.ModelCard)
    ] = model_card_pb2.ModelCard

    def to_json(self) -> str:
        """Write ModelCard to JSON."""
        model_card_dict = self.to_dict()
        model_card_dict[_SCHEMA_VERSION_STRING] = validation.get_latest_schema_version()
        return json_lib.dumps(model_card_dict, indent=2)

    def from_json(self, json_dict: Dict[str, Any]) -> None:
        """Reads ModelCard from JSON.

        If ModelCard fields have already been set, this function will overwrite any
        existing values.

        Args:
          json_dict: A JSON dict from which to populate fields in the model card
            schema.

        Raises:
          JSONDecodeError: If `json_dict` is not a valid JSON string.
          ValidationError: If `json_dict` does not follow the model card JSON
            schema.
          ValueError: If `json_dict` contains a value not in the class or schema
            definition.
        """

        def _populate_from_json(
            json_dict: Dict[str, Any], field: BaseModelCardField
        ) -> BaseModelCardField:
            for subfield_key in json_dict:
                if subfield_key.startswith(_SCHEMA_VERSION_STRING):
                    continue
                elif not hasattr(field, subfield_key):
                    raise ValueError(
                        "BaseModelCardField %s has no such field named '%s.'"
                        % (field, subfield_key)
                    )
                elif isinstance(json_dict[subfield_key], dict):
                    subfield_value = _populate_from_json(
                        json_dict[subfield_key], getattr(field, subfield_key)
                    )
                elif isinstance(json_dict[subfield_key], list):
                    subfield_value = []
                    for item in json_dict[subfield_key]:
                        if isinstance(item, dict):
                            new_object = field.__annotations__[subfield_key].__args__[
                                0
                            ]()  # pytype: disable=attribute-error
                            subfield_value.append(_populate_from_json(item, new_object))
                        else:  # if primitive
                            subfield_value.append(item)
                else:
                    subfield_value = json_dict[subfield_key]
                setattr(field, subfield_key, subfield_value)
            return field

        validation.validate_json_schema(json_dict)
        self.clear()
        _populate_from_json(json_dict, self)

    @staticmethod
    def _get_reports_results(reports: List) -> Counter:
        """Get summary of tests passed and failed across multiple reports.
        Each report has a list of tests.

        Args:
          reports: List of reports to calculate over.

        Returns:
          Counter of test cases passed and failed
        """
        result_counter = Counter()

        for r in reports:
            tests = r.tests
            passed = sum(1 for t in tests if t.passed)
            result_counter.update({"passed": passed, "failed": len(tests) - passed})

        return dict(result_counter)

    def get_test_results(self) -> Dict[str, Counter]:
        """Return overall number of tests passed and failed across performance metrics,
        explainability reports, fairness reports.

        Returns:
          Counter of test cases passed and failed for performance tests, explainability tests,
          and fairness tests in a dictionary.
        """
        performance_metrics = self.quantitative_analysis.performance_metrics
        explainability_reports = self.explainability_analysis.explainability_reports
        fairness_reports = self.fairness_analysis.fairness_reports

        return {
            "performance_tests": self._get_reports_results(performance_metrics),
            "explainability_tests": self._get_reports_results(explainability_reports),
            "fairness_tests": self._get_reports_results(fairness_reports),
        }
