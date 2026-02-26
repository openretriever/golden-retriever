# Retriever Distribution

## Installation

### Option 1: Using Pixi (Recommended)

This distribution contains a `pixi.toml` configured to install the included wheel and dependencies.

1.  Install Pixi: `curl -fsSL https://pixi.sh/install.sh | bash`
2.  Install environment:
    ```bash
    pixi install
    ```
3.  Run examples:
    ```bash
    pixi run python examples/tutorial/009_dora_perception.py
    ```
4.  Enter shell:
    ```bash
    pixi shell
    ```

5.  Interactive Python (IPython):
    ```bash
    pixi run ipython
    # Then: import retriever
    ```

### Option 2: Standard Pip

1.  Create a virtual environment (optional).
2.  Install the wheel:
   ```bash
   pip install install/retriever-0.0.0-py3-none-any.whl
   ```
3.  Install example dependencies manually (numpy, opencv-python, etc).

## Running Examples

The `examples/` directory contains tutorial and advanced examples.

```bash
# Via Pixi
pixi run python examples/tutorial/009_dora_perception.py

# Or
pixi run demo-webcam-detection

# Via Pip environment
python examples/tutorial/009_dora_perception.py
```
