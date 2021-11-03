#!/usr/bin/env bash

# exit script upon error
set -e

# variables
DOCS=build/docs
NB=$DOCS/content/fairness/credit-card-fraud-example-notebook

# build the doc files into a `build/docs` folder
printf "\nBuilding doc files...\n"
pydoc-markdown

# edit file extensions for notebook files
printf "\nEditing file extensions for Jupyter Notebooks...\n"
mv $NB.md $NB.ipynb

# edit mkdocs.yml to reference the notebook files
printf "\nEditing mkdocs.yml to point to notebooks...\n"
python3.8 convert_md_to_ipynb.py

# serve or build HTML files, depending on argument $1
case $1 in

  serve)
    printf "\nServing docs locally...\n\n"
    mkdocs serve -f $DOCS/mkdocs.yml
    ;;

  build)
    printf "\nBuilding HTML docs...\n\n"
    mkdocs build -f $DOCS/mkdocs.yml -d ../html
    ;;

  *)
    printf "\nERROR: Unrecognised argument. Use './docs.sh serve' or './docs.sh build.'\n"
    ;;
esac