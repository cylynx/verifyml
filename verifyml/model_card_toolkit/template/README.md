# Model Card Templates

This section is adapted from Google's [Model Card templates documentation](https://github.com/tensorflow/model-card-toolkit/blob/master/model_card_toolkit/documentation/guide/templates.md).

This folder contains the Jinja templates used to render Model Cards. There is a `html` subfolder and `md` subfolder, for HTML and Markdown templates respectively.

Feel free to modify them, but it is recommended to create new ones that are relevant to your use case.

## Sample Usage

This sample demonstrates how you can use templates to render your Model Card.

```python
from verifyml.model_card_toolkit import ModelCardToolkit

# set output directory for model card assets, relative to your current location
output_dir = YOUR_OUTPUT_DIR
mct = ModelCardToolkit(output_dir=output_dir)

# init model card and copy contents of verifyml/model_card_toolkit/template/
# into ./output_dir/template/
mc = mct.scaffold_assets()

# set model card fields
...

# update the model card with these fields
mct.update_model_card_json(mc)

# METHOD 1: use defaults for template_path and output_file
#   template_path = ./output_dir/template/html/default_template.html.jinja
#   output_file = 'model_card.html'
mct.export_format()

# METHOD 2: specify the template to use and desired output file name, then export as html
template_path = YOUR_TEMPLATE_PATH
output_file = YOUR_MODEL_CARD_NAME.html
mct.export_format(template_path=template_path, output_file=output_file)
```

If you tweaked the [default template](verifyml/model_card_toolkit/template/html/default_template.html.jinja), `METHOD 1` above would be sufficient.

If you created your own template, use `METHOD 2`.
