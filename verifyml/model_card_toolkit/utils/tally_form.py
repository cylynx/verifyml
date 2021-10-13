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

import json
from .. import model_card_toolkit as mct
from .. import model_card
from typing import List, Union


def get_answer(questions: str, label: str) -> Union[str, None]:
    """Parse tally form by type and return form field value"""

    match = next(filter(lambda d: d["label"] == label, questions))
    if match["type"] in ["INPUT_TEXT", "TEXTAREA", "INPUT_EMAIL"]:
        return match["value"]
    elif match["type"] == "MULTIPLE_CHOICE":
        result = next(filter(lambda d: d["id"] in match["value"], match["options"]))
        return result["text"]
    elif match["type"] == "CHECKBOXES":
        value_list = match["value"]
        return [
            result["text"]
            for result in filter(lambda d: d["id"] in value_list, match["options"])
        ]
    else:
        raise ValueError(f"Processing logic for {match['type']} not defined")


def str_to_list(text: Union[str, None]) -> Union[List, None]:
    """Split string by comma and return list"""

    return text.split(",") if isinstance(text, str) else text


def tally_form_to_mc(form_path: str):
    """Parse tally json response and automates creation of model card.

    Should use in combination with the tally form utility which helps
    bootstrap a model card.

    Args:
      form_path: path to tally json response.

    Returns:
      A ModelCard data object seralized to string.
    """

    with open(form_path, "r") as myfile:
        data = myfile.read()

    res = json.loads(data)
    questions = res["data"]["fields"]
    mc = mct.ModelCard()

    ## Parse model details
    mc.model_details.name = get_answer(questions, "Name")
    mc.model_details.overview = get_answer(questions, "Overview")
    mc.considerations.users = [
        model_card.User(description=get_answer(questions, "Intended Users"))
    ]
    mc.considerations.use_cases = [
        model_card.UseCase(description=get_answer(questions, "Intended Use Cases"))
    ]
    mc.model_details.version = model_card.Version(name=get_answer(questions, "Version"))
    mc.model_details.owners = [
        model_card.Owner(
            name=get_answer(questions, "Product Owner(s)"), role="Product Owner(s)"
        ),
        model_card.Owner(
            name=get_answer(questions, "Model Developer(s)"), role="Model Developer(s)"
        ),
        model_card.Owner(name=get_answer(questions, "Reviewer(s)"), role="Reviewer(s)"),
    ]
    mc.model_details.regulatory_requirements = " ".join(
        get_answer(
            questions,
            "Please select any regulatory guidelines which the model should be in compliance with",
        )
    )

    # Parse data
    mc.model_parameters.data = [
        model_card.Dataset(
            name=get_answer(questions, "Name of dataset"),
            description=get_answer(questions, "Description of dataset"),
            sensitive=model_card.SensitiveData(
                sensitive_data=str_to_list(
                    get_answer(questions, "Protected attributes in dataset")
                ),
                sensitive_data_used=str_to_list(
                    get_answer(questions, "Protected attributes in production")
                ),
                justification=get_answer(
                    questions, "Justification of use of protected attributes"
                ),
            ),
        )
    ]

    if get_answer(questions, "Name of dataset 2"):
        mc.model_parameters.data.append(
            model_card.Dataset(
                name=get_answer(questions, "Name of dataset 2"),
                description=get_answer(questions, "Description of dataset 2"),
                sensitive=model_card.SensitiveData(
                    sensitive_data=str_to_list(
                        get_answer(questions, "Protected attributes in dataset 2")
                    ),
                    sensitive_data_used=str_to_list(
                        get_answer(
                            questions, "Protected attributes in production (dataset 2)"
                        )
                    ),
                    justification=get_answer(
                        questions,
                        "Justification of use of protected attributes (dataset 2)",
                    ),
                ),
            )
        )

    if get_answer(questions, "Name of dataset 3"):
        mc.model_parameters.data.append(
            model_card.Dataset(
                name=get_answer(questions, "Name of dataset 3"),
                description=get_answer(questions, "Description of dataset 3"),
                sensitive=model_card.SensitiveData(
                    sensitive_data=str_to_list(
                        get_answer(questions, "Protected attributes in dataset 3")
                    ),
                    sensitive_data_used=str_to_list(
                        get_answer(
                            questions, "Protected attributes in production (dataset 3)"
                        )
                    ),
                    justification=get_answer(
                        questions,
                        "Justification of use of protected attributes (dataset 3)",
                    ),
                ),
            )
        )

    ## Quantitative analysis
    mc.quantitative_analysis.performance_metrics = [
        model_card.PerformanceMetric(
            type=get_answer(
                questions,
                "What is the key metric used in evaluating the model's performance?",
            ),
        )
    ]

    if get_answer(questions, "2nd metric (if applicable)"):
        mc.quantitative_analysis.performance_metrics.append(
            model_card.PerformanceMetric(
                type=get_answer(questions, "2nd metric (if applicable)"),
            )
        )

    if get_answer(questions, "3rd metric (if applicable)"):
        mc.quantitative_analysis.performance_metrics.append(
            model_card.PerformanceMetric(
                type=get_answer(questions, "3rd metric (if applicable)"),
            )
        )

    ## Fairness consideration
    mc.considerations.fairness_assessment = [
        model_card.FairnessAssessment(
            group_at_risk=get_answer(
                questions,
                "Who are the individuals and groups that are considered to be at-risk of being systematically disadvantaged by the system?",
            ),
            benefits=get_answer(questions, "Expected Benefits"),
            harms=get_answer(questions, "Expected Harms"),
            mitigation_strategy=get_answer(questions, "Mitigation strategies"),
        )
    ]

    ## Fairness analysis
    mc.fairness_analysis.fairness_reports = [
        model_card.FairnessReport(
            type=get_answer(questions, "Type of fairness analysis conducted"),
            segment=get_answer(questions, "Segment of analysis"),
            description=get_answer(questions, "Description of fairness analysis"),
        )
    ]

    if get_answer(questions, "Type of fairness analysis conducted 2"):
        mc.fairness_analysis.fairness_reports.append(
            model_card.FairnessReport(
                type=get_answer(questions, "Type of fairness analysis conducted 2"),
                segment=get_answer(questions, "Segment of analysis 2"),
                description=get_answer(questions, "Description of fairness analysis 2"),
            )
        )

    if get_answer(questions, "Type of fairness analysis conducted 3"):
        mc.fairness_analysis.fairness_reports.append(
            model_card.FairnessReport(
                type=get_answer(questions, "Type of fairness analysis conducted 3"),
                segment=get_answer(questions, "Segment of analysis 3"),
                description=get_answer(questions, "Description of fairness analysis 3"),
            )
        )

    return mc.to_proto()
