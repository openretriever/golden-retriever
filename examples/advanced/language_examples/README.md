# Language Examples

These concise advanced examples teach one narrow rule: keep primitive language
payloads canonical in core, then compose them structurally with perception or
planning outputs.

1. `caption_to_plan.py`: turn a short `Caption` into a tiny `PlanText`.
2. `grounded_reference.py`: combine a `ReferringExpression` with a `DetectionBatch`
   to emit a `GroundedPhrase`.

Run them with the local editable core so they use the current Retriever type
surface:

```bash
pixi run -e golden-local demo-language-caption-plan
pixi run -e golden-local demo-language-grounded-reference
```
