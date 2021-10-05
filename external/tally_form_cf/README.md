# Model Ops Card Builder

Bootstrap a model card by filling up the following [web form](https://tally.so/r/mR4Nlw).  

The response in JSON format will be sent to your email.

To parse the form response to a model card format, use the `tally_form_to_mc` utils function.

```py
from model_card_toolkit.utils.tally_form import tally_form_to_mc

# Convert form response to model card protobuf
pb = tally_form_to_mc("sample-form-response.json")
```

For more details check out `examples/tally_form_example.ipynb`.

## Deploy

Form response is sent via a webhook to a Google Cloud Function.  

`main.py` contains the script to compose an email with the response and send it to the user's specified address using mailersend service.

To deploy the cloud function, authenticate with `gcloud auth login` and run `deploy.sh`. A valid `MAILERSEND_API_KEY` environment variable has to be present and will be used as part of deployment.
