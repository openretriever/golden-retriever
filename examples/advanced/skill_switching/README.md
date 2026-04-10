# Skill Switching with Retriever

Two versions of the same skill-switching controller: a classic explicit-wiring variant and a fan-in variant.

## Quick Start

```bash
pixi run demo-skill-switching
pixi run demo-skill-switching-fanin
```

## Direct commands

```bash
pixi run python -m examples.advanced.skill_switching.main
pixi run python -m examples.advanced.skill_switching.main --fan-in
```

## What it demonstrates

- mode switching without stale actions
- explicit inactive-skill signaling
- classic explicit wiring versus a single fan-in packet port
- a small closed-loop coordination graph that is still easy to inspect
