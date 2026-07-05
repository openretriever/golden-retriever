# Language and Grounding v1

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever.org/start/">Start path</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Visual quickstart</a>
  <a href="/">Golden overview</a>
  <a href="../llms.txt">Golden agent map</a>
</div>


This guide adds one small language-facing ladder on top of the shared primitive
Retriever type families.

The rule stays the same as the rest of the advanced examples:
- reuse generic standard payloads from core when they already fit,
- keep robot-facing planning bundles in Golden or Hub type packs,
- use composite `Flow[...]` structure for local grouping,
- keep model-specific request/response packets out of the first teaching path.

## 1. Caption to primitive plan text

Start with the smallest language-only example:

```bash
pixi run -e golden-local demo-language-caption-plan
```

This uses the canonical language primitives directly:
- `Caption`
- `PlanStepText`
- `PlanText`

It demonstrates the preferred surface for simple planner outputs: primitive
plan text first, larger domain bundles later if they prove stable.

## 2. Ground a referring expression with detections

Then add one structural composition example:

```bash
pixi run -e golden-local demo-language-grounded-reference
```

This combines:
- `ReferringExpression`
- `DetectionBatch`
- `GroundedPhrase`

The interesting part is the structure, not a custom envelope. The example keeps
that explicit by using language and perception primitives directly.

## 3. Relationship to the other ladders

- `perception_examples/` teaches frame, detection, mask, and point-target payloads.
- `memory_examples/` adds persistent local state on top of those primitives.
- `language_examples/` adds primitive language and grounding payloads.
- `core_composition/` then shows how to compose larger reusable pipeline slices.

## 4. What stays out of the first ladder

These examples intentionally do **not** teach:
- model-specific VLM request/response packets,
- prompt orchestration metadata,
- large domain plans,
- larger integrated planning bundles.

Those belong in higher-level packages or later examples once the primitive type
surface is already clear.
