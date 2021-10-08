# Publishing to PyPI

- Set version number and configs in `setup.cfg`

```bash
pip install --upgrade setuptools build twine

# build package files
python -m build

# upload to testpypi
python -m twine upload --repository testpypi dist/*
```
