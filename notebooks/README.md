# Notebooks

This repo keeps notebooks git-friendly by treating percent-format Python files as
the source of truth.

## Layout

- `notebooks/src/*.py`: source notebooks in `py:percent` format
- `notebooks/generated/*.ipynb`: generated notebook files for UI-based execution

## Build the notebooks

```bash
pixi run notebook-to-ipynb-demo
pixi run notebook-to-ipynb-hub
```

Those commands regenerate:

- `notebooks/generated/retriever_demo.ipynb`
- `notebooks/generated/hub_demo.ipynb`

`retriever_demo.ipynb` stays a small Jupytext mechanics demo for the packaged Golden environment.

`hub_demo.ipynb` is the Hub-first notebook. Run it from the `golden-local` environment so the Golden example feature set is available:

```bash
pixi install -e golden-local
pixi run -e golden-local demo-hub-notebook-source
```

The Hub notebook does not hardcode any private or organization-specific module refs. Set these environment variables before running it if you want the live Hub cells to execute:

- `RETRIEVER_HUB_HELLO_WORLD_MODULE`
- `RETRIEVER_HUB_COMPOSE_MODULE`

Use the `.py` files for review in git. Regenerate the `.ipynb` files only when you need a notebook UI such as Jupyter or Deepnote.
