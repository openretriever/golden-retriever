# Notebooks

This repo keeps notebooks git-friendly by treating percent-format Python files as
the source of truth.

## Layout

- `notebooks/src/*.py`: source notebooks in `py:percent` format
- `notebooks/generated/*.ipynb`: generated notebook files for UI-based execution

## Build the demo notebook

```bash
pixi run notebook-to-ipynb-demo
```

That command regenerates:

- `notebooks/generated/retriever_demo.ipynb`

This notebook is a Jupytext mechanics demo for the current Golden environment. It does not currently demonstrate Retriever Hub, because the bundled `retriever_dist` snapshot in this repo does not yet expose Hub.

Use the `.py` file for review in git. Regenerate the `.ipynb` only when you
need a notebook UI such as Jupyter or Deepnote.


If you want notebook code to run against the live sibling `retriever-mirror` checkout instead of the bundled wheel, use the `golden-local` environment described in the repo root README. The current notebook content remains a mechanics demo either way.
