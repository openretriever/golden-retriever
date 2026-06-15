# MuJoCo Web Manipulation

This example runs the MuJoCo physics simulation in the browser through the
`mujoco-js` WebAssembly package. Retriever runs on the Python side as a 50 Hz
controller connected over a WebSocket:

```text
Browser MuJoCo WASM -> BrowserStateSource -> RetrieverControllerFlow -> WebControlSink -> Browser MuJoCo WASM
```

## Usage

```bash
pixi run demo-mujoco-web
```

Then open:

```text
http://127.0.0.1:8010
```

The demo intentionally uses Retriever's `in-process` backend so the FastAPI
WebSocket bridge and the Retriever flows share state directly.
