# Tests

Files to test the functionality of the FEAT test classes using `pytest`.

## Installation

```bash
pip install -U pytest
```

## Running

Run this command while in this directory:

```bash
# -v: verbose
pytest -v
```

Otherwise, specify it like this:

```bash
pytest <PATH TO THIS TEST DIR>/ -v
```

## Configuration

Use the `[tool.pytest.ini_options]` section of VerifyML's [`pyproject.toml` file](https://github.com/cylynx/verifyml/blob/main/pyproject.toml) to configure `pytest`.
