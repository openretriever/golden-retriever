# Multi-Agent Communication Example

A coordination demo that shows task announcements, bidding, assignment, and execution monitoring across multiple flows.

## Quick Start

```bash
pixi run demo-multi-agent-communication
```

## Other modes

```bash
pixi run python examples/advanced/multi_agent_communication/app.py --mode step --steps 60 --dt 0.1
pixi run python examples/advanced/multi_agent_communication/app.py --mode run --backend multiprocessing --duration 6
pixi run python examples/advanced/multi_agent_communication/app.py --mode run --backend dora --duration 6
```

## What it demonstrates

- one task source broadcasting work announcements
- multiple bidding agents producing candidate assignments
- a fan-in auctioneer selecting one assignee
- downstream progress reports and a monitoring sink

Use the stepper or `multiprocessing` path first. The `dora` mode is there when you specifically want the distributed runtime split.
