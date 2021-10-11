If the [provided model tests](verifyml/model_tests) are insufficient, you can easily create your own by extending the [ModelTest](verifyml/model_tests/ModelTest.py) abstract base class.

When creating a subclass from ModelTest, there are a few requirements for the test to render properly on a Model Card:

1. It must have a `run()` method. The method should do these things:
   a. Run your test and save the results into `self.result`
   b. Update `self.passed`, a boolean indicating if the test result passes your defined condition
2. If plots are to be displayed, the subclass also needs a method (e.g. `plot()`) that stores them in `self.plots` as base64-encoded strings
3. Optionally, the subclass can overwrite the base class' `test_name` and `test_desc` to display that information in the model card as well

To see a toy example test, refer to `DEVELOPMENT.ipynb`.
