# 07 — Resource Management (Tutorial)

Retriever’s runtime has **resource hints** that can influence execution grouping (fusion).

Today, the runtime supports:
- `FlowConfig.priority` (co-location constraint for strict grouping)
- `FlowConfig.cpu_affinity` (must overlap to co-locate under strict grouping)

The current compiler exposes this as a **policy**:
- `policy="aggressive"` (default): ignores resource compatibility
- `policy="strict"`: enforces resource compatibility

## Run

```bash
pixi run python -m examples.tutorial.07_resource_management.00_strict_resource_fusion --case compatible
pixi run python -m examples.tutorial.07_resource_management.00_strict_resource_fusion --case priority_mismatch
pixi run python -m examples.tutorial.07_resource_management.00_strict_resource_fusion --case affinity_mismatch
pixi run python -m examples.tutorial.07_resource_management.01_resource_hints --print-ir
```
