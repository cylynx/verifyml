# Fairness Assessment 

Cylynx's VerifyML fairness assessment determines whether outcomes provided by your model are fair across the specified sensitive attributes. Depending on the context, examples of sensitive attributes can be sex, ethnicity and age. 

In our assessment, we will identify and flag subgroups that a model biases against, based on these options:
- type of test (e.g. disparity test, feature importance test)
- fairness metric of interest (e.g. false postive rate, false negative rate)
- method (e.g. ratio, chi-sq test)
- threshold value

End-users will have to provide with their own ML model and specify the above arguments using a Python interface, after which the results will be compiled into a model scorecard through the model card toolkit. Fairness assessment is one of the segments of the scorecard. The tookit also provides trade-off analysis between business objectives and fairness objectives across different models.

## Fairness Tests and Explainability 

Currently, we provide 5 different tests. Each test comes with an outcome, being passed or failed, and respective graphs for visualisation.

#### 1) Subgroup Disparity Test
This test checks if there is disparity (over a specified threshold) in the fairness metrics between the best and worst performing subgroups of the sensitive attribute. Choices for method of disparity measure includes ratio, difference and chi-square test. The test ensures one group is not given too much privilege/handicap over another. Depending on the context and justifications, test arguments like threshold and fairness metric have to be carefully selected by the end-user. 

#### 2) Min/Max Metric Threshold Test
This test checks if the fairness metric of the subgroups passes the mininum/maximum threshold specified. For example, a reasonably fair maximum threshold for false positive rate in the case of fraud detection can be 2.5%, where any value greater can be detrimental to the business and is best avoided. In contrast to the above disparity test, disparity among the subgroups are not considered in this test as long as their metrics pass the threshold. This is crucial as it may not be ethical and justified to worsen one of the subgroup's metric to reduce the fairness disparity among subgroups. ROC curve will be plotted to visualise the trade-off between business and fairness objective.

#### 3) Perturbation Test
This test checks if the fairness metrics of the subgroups in the original data are not worse than that of the perturbed data by a specified threshold. In the perturbed data, the values in the sensitive attribute column will be randomly shuffled while keeping all other features unchanged. This renders the attribute insignificant in explaining the model, providing a what-if scenario on the fairness metrics performance if the sensitive attribute were to be completely removed from the model. 

#### 4) Feature Importance Test
This test checks if the subgroups of the specified sensitive attributes are the model's top influential features based on user-selected or [Shapely](https://christophm.github.io/interpretable-ml-book/shapley.html) importance values. A sensitive subgroup as the model's top feature is not entirely desirable due to the model's over-reliance on that feature to make predictions, potentially imposing ethical bias among the subgroups. One solution to this is to reduce dependence on the sensitive attribute by introducing other features (into the model) that is not only correlated to the sensitive attribute but also has a more causal relation with the outcome.

#### 5) Data Shift Test
This test checks if there is a shift in the distribution of the sensitive attributes between the training data and evaluation data. As part of continuous model deployment, this provides alerts to end-users on the possibility of model performance degradation (based on both fairness and business objectives) and that the model requires retraining on newer data.


## Example Notebooks

[Credit card fraud example](../examples/credit_card_fraud_example.ipynb)

