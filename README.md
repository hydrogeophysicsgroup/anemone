# anEMone
a pytorch based 1D electromagnetic modelling and inversion code.

## Installation

```console
pip install anemone-core
```

The package is published on PyPI as `anemone-core`, but is imported as `anemone`:

```python
import anemone
```

## Documentation

The documentation site is built from markdown files in `docs/` using MkDocs.

Install the documentation tools:

```console
pip install -e ".[docs]"
```

Preview the site locally:

```console
mkdocs serve
```

Build the static site:

```console
mkdocs build
```

Jupyter notebooks can be rendered as documentation pages through `mkdocs-jupyter` when curated notebooks are placed under `docs/` and added to `mkdocs.yml`.
