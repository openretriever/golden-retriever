# Contributing

GoldenRetriever is the companion examples and system-integration repository for the core Retriever runtime.

Keep contributions small and example-first:

- update docs when launch commands, environments, or example behavior changes;
- keep core runtime API changes in the main `retriever` repository;
- keep hardware, model, and dataset dependencies behind optional Pixi features;
- avoid committing generated logs, notebooks, recordings, credentials, or local machine paths;
- prefer concise, runnable examples over broad framework expansion.

## Setup

```bash
pixi install
pixi run -e docs docs-build
```

For Golden example launch commands that need the companion runtime package:

```bash
pixi install -e golden-local
pixi run -e golden-local demo-perception-detection-flow
```

Golden example environments should import the runtime as `retriever`; update docs when the packaged runtime dependency changes.

## Pull Requests

Include:

- the user-facing behavior changed;
- the exact commands run;
- any optional hardware/model dependency required;
- screenshots or short logs for visual demos when relevant.

Do not include private notes, private repository names, local filesystem paths, credentials, or unpublished model/data artifacts.
