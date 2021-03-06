#!/usr/bin/python

""" 
Small hack to edit `build/docs/mkdocs.yml` and change a jupyter notebook's
file extension from .md (auto-generated when building docs) to .ipynb

Changing it allows jupyter notebooks to be rendered, thanks to the
mkdocs-jupyter plugin.
"""
import yaml

# relative from root, autogenerated by pydoc-markdown
MKDOCS_YML_PATH = "build/docs/mkdocs.yml"

# open file
with open(MKDOCS_YML_PATH) as f:
    y = yaml.safe_load(f)

# update file extension for credit card fraud example notebook entry
curr = y["nav"][1]["Fairness"][1]["Credit Card Fraud Example Notebook"]

if curr.endswith(".ipynb"):
    print("File already in notebook format")
    pass
elif curr.endswith(".md"):
    y["nav"][1]["Fairness"][1]["Credit Card Fraud Example Notebook"] = (
        curr[:-3] + ".ipynb"
    )

    # write it back to the file
    with open(MKDOCS_YML_PATH, "w") as f:
        yaml.safe_dump(y, f, default_flow_style=False)
else:
    print("Unrecognised file extension")
