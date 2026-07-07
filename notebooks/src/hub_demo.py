# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: "1.3"
# --

# %% [markdown]
# # Retriever Hub notebook demo
#
# This notebook is Hub-first and assumes you are running it from Golden's
# `golden-retriever` environment so the Golden example feature set is available.
#
# Configure one or both environment variables before running the source file or
# launching the generated notebook:
#
# - `RETRIEVER_HUB_HELLO_WORLD_MODULE`: module ref that exports `GreeterFlow`,
#   `GreeterInput`, and `GreeterOutput`
# - `RETRIEVER_HUB_COMPOSE_MODULE`: module ref that exports one or more
#   `Build*Pipeline` / `Build*PipelineFlow` factories
#
# The notebook is still useful without them: it validates the runtime setup
# and shows the expected Hub surface.

# %%
import os
from pprint import pprint

import retriever
from retriever import hub

print('retriever:', retriever.__file__)
print('hub:', hub.__file__)
print('version:', getattr(retriever, '__version__', '<unknown>'))


# %%
def public_exports(module_proxy):
    return sorted(name for name in dir(module_proxy) if not name.startswith('_'))


hello_ref = os.environ.get('RETRIEVER_HUB_HELLO_WORLD_MODULE', '').strip()
compose_ref = os.environ.get('RETRIEVER_HUB_COMPOSE_MODULE', '').strip()

print('RETRIEVER_HUB_HELLO_WORLD_MODULE =', hello_ref or '<unset>')
print('RETRIEVER_HUB_COMPOSE_MODULE =', compose_ref or '<unset>')


# %% [markdown]
# ## 1. Whole-module import and explicit export import
#
# This is the smallest Hub story: load a published module, inspect its exports,
# then pull out one explicit flow/type boundary and execute it locally with
# `step(...)`.

# %%
if not hello_ref:
    print(
        "Set RETRIEVER_HUB_HELLO_WORLD_MODULE to a published module ref, "
        "for example 'your-org/hello-world', then rerun this cell."
    )
else:
    try:
        hello_module = hub.use(hello_ref)
    except Exception as exc:
        print(f'failed to load {hello_ref!r}: {exc}')
    else:
        print('whole-module exports:')
        pprint(public_exports(hello_module))

        try:
            GreeterFlow = hub.use(f'{hello_ref}:GreeterFlow')
            GreeterInput = hub.use(f'{hello_ref}:GreeterInput')
            GreeterOutput = hub.use(f'{hello_ref}:GreeterOutput')
        except Exception as exc:
            print(f'failed to load explicit exports from {hello_ref!r}: {exc}')
        else:
            greeter = GreeterFlow(prefix='Hi')
            result = greeter.step(GreeterInput(name='Retriever'))
            print('explicit export result:', result)
            print('is GreeterOutput:', isinstance(result, GreeterOutput))


# %% [markdown]
# ## 2. Inspect composable pipeline exports
#
# This section does not assume one fixed module shape. It looks for exported
# pipeline builders and pipeline-flow builders and exercises them when the
# signatures are zero-argument.

# %%
if not compose_ref:
    print(
        'Set RETRIEVER_HUB_COMPOSE_MODULE to a published module ref that '
        'exports Build*Pipeline / Build*PipelineFlow factories, then rerun this cell.'
    )
else:
    try:
        compose_module = hub.use(compose_ref)
    except Exception as exc:
        print(f'failed to load {compose_ref!r}: {exc}')
    else:
        exports = public_exports(compose_module)
        print('compose-module exports:')
        pprint(exports)

        pipeline_builders = [
            name for name in exports if name.startswith('Build') and name.endswith('Pipeline')
        ]
        pipeline_flow_builders = [
            name for name in exports if name.startswith('Build') and name.endswith('PipelineFlow')
        ]

        print('pipeline builders:', pipeline_builders or '<none>')
        print('pipeline-flow builders:', pipeline_flow_builders or '<none>')

        if pipeline_builders:
            builder = getattr(compose_module, pipeline_builders[0])
            try:
                pipe = builder()
            except TypeError as exc:
                print(
                    f'skipping live pipeline build: {pipeline_builders[0]} '
                    f'requires arguments ({exc})'
                )
            else:
                print('live pipeline node ids:', sorted(pipe.get_flow_dict().keys()))

        if pipeline_flow_builders:
            flow_builder = getattr(compose_module, pipeline_flow_builders[0])
            try:
                wrapped = flow_builder()
            except TypeError as exc:
                print(
                    f'skipping wrapped pipeline build: {pipeline_flow_builders[0]} '
                    f'requires arguments ({exc})'
                )
            else:
                print('wrapped pipeline flow object:', wrapped)
                print('wrapped pipeline flow class:', type(wrapped).__name__)
