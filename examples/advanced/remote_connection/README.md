# Remote Connection Example

This example demonstrates how to distribute a Retriever pipeline across multiple computers using the **Dora** backend.

## Scenario

-   **Machine A (Local)**: Runs the `Controller` (simulated robot driver). It requires low-latency hardware access.
-   **Machine B (Remote)**: Runs the `Policy` (AI model). It requires heavy compute (e.g., GPU) and might be on a separate server.

## Prerequisites

1.  **Dora installed** on both machines.
2.  **Retriever installed** on both machines.
3.  **Network connectivity** between machines.

## Setup

### 1. Configure Dora Coordinator

On one machine (e.g., Machine A), create a `coordinator.yaml` describing the cluster:

```yaml
# coordinator.yaml
dora_version: 0.3.5

port: 53290 # Control port

# Define available machines (daemon nodes)
dependents:
  # Machine A (Local)
  machine_a:
    ip: 127.0.0.1  # Or actual IP
    # ... SSH credentials if needed to auto-start ...

  # Machine B (Remote)
  machine_b:
    ip: 192.168.1.50 # Replace with actual IP of Machine B
    # ...
```

*Note: For manual daemon startup (simplest), you don't need complex SSH config. Just ensure `dora daemon` is running on each node.*

### 2. Start Dora Daemons

**On Machine A:**
```bash
# Start coordinator
dora coordinator --config coordinator.yaml

# In another terminal, start daemon for Machine A
dora daemon --machine-id machine_a --coordinator-addr 127.0.0.1:53290
```

**On Machine B:**
```bash
# Start daemon for Machine B
dora daemon --machine-id machine_b --coordinator-addr <IP_OF_MACHINE_A>:53290
```

## Running the Application

The `app.py` script defines the pipeline and assigns deployments:

```python
    # Option 1: Static Deployment
    # controller.deploy("machine_a")
    # policy.deploy("machine_b")
    
    # Option 2: Runtime Deployment (Flexible)
    # See pipe.run(...) arguments

```

Run the application (from Machine A):

```bash
# This will compile the graph, send it to the coordinator, 
# which dispatches nodes to respective daemons.
python examples/advanced/remote_connection/app.py --machine-a machine_a --machine-b machine_b
```

## How It Works

1. **Deployment**:
   - **Static**: `Flow.deploy(machine)` tags the node properties.
   - **Runtime (Preferred)**: `pipe.run(deploy={...})` passes dynamic overrides at execution time.
2. **Compilation**: The compiler translates these into `_unstable_deploy` in Dora YAML.
3. **Execution**: The Dora Coordinator schedules nodes based on these tags.


