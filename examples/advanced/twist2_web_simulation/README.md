# TWIST2 MuJoCo Web

Runs the Unitree G1 TWIST2 demo with MuJoCo physics in the browser through
`mujoco-js`, while Retriever runs the motion stream and ONNX policy in Python.

```bash
pixi run -e torch demo-twist2-web
```

Open the printed local URL. The browser sends `qpos`/`qvel` over a WebSocket;
Retriever reconstructs proprioception, runs `Twist2PolicyFlow`, and returns
policy actions that the browser applies through the same PD constants as the
native demo.

