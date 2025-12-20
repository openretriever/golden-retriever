# Planning Pipeline Examples

This directory contains three complete planning pipeline examples that demonstrate different architectural approaches in the Retriever framework. These examples showcase the evolution from traditional robotics code to modern, composable pipeline architectures.

## 📁 Example Files

### 1. `example_oracle.py` - Original Implementation
The baseline example showing traditional robotics pipeline structure:
- Direct function calls and procedural execution
- Manual state management and error handling
- Tightly coupled components
- Minimal type safety

**Use this when**: Learning the basic concepts or working with legacy code.

### 2. `example_oracle_typed.py` - Typed Architecture
Enhanced version with strong typing and structured data:
- Comprehensive type definitions for all data structures
- Type-safe interfaces between components
- Better error handling and debugging
- Structured execution status and metadata

**Use this when**: You want type safety but aren't ready for full Flow composition.

### 3. `example_oracle_flow.py` - Flow-Based Pipeline ⭐
Modern composable architecture using the Retriever Flow framework:
- **Module Protocol**: All components implement `Module[I, O]` for type-safe composition
- **Flow Combinators**: Use `.then()` and `.fanout()` for pipeline construction
- **Eff Monad**: Clean handling of stateful robot operations
- **Parallel Processing**: Multi-sensor fusion with `fanout()`
- **Reusable Components**: Easy testing and cross-robot compatibility

**Use this when**: Building production robotics systems that need to scale and evolve.

## 🏗️ Architecture Comparison

| Feature | Original | Typed | Flow-Based |
|---------|----------|-------|------------|
| Type Safety | ❌ | ✅ | ✅ |
| Composability | ❌ | ⚠️ | ✅ |
| Testability | ❌ | ⚠️ | ✅ |
| Reusability | ❌ | ⚠️ | ✅ |
| Parallel Processing | ❌ | ❌ | ✅ |
| State Management | Manual | Manual | Eff Monad |
| Error Handling | Basic | Structured | Comprehensive |

## 🎯 Key Concepts Demonstrated

### Bilevel Planning Architecture
All examples demonstrate the core bilevel planning pattern:
1. **High-level Planning**: VLM-based task decomposition
2. **Low-level Execution**: Oracle agent skill execution

### Pipeline Components
Common components across all examples:
- **Environment Observation**: Multi-camera sensor data
- **VLM Planning**: GPT-4V for high-level reasoning
- **Plan Formatting**: Structured plan representation
- **Robot Execution**: Oracle-based skill execution
- **Result Tracking**: Success metrics and metadata

### Foundation Model Integration
- Ray-based distributed inference
- LangSAM for object segmentation
- OpenAI GPT-4V for visual planning
- Type-safe actor handle management

## 🚀 Flow-Based Pipeline Patterns

The Flow example showcases key architectural patterns:

### Sequential Composition
```python
pipeline = (
    extract_images
    .then(save_images)
    .then(call_vlm)
    .then(format_plan)
)
```

### Parallel Processing
```python
# Multi-sensor fusion
stereo_vision = left_camera.fanout(right_camera)

# Multi-hypothesis planning
multi_plan = rrt_planner.fanout(prm_planner)
```

### Stateful Operations with Eff Monad
```python
def execute_step(instruction: str) -> Eff[RobotState, bool]:
    def run_step(state: RobotState) -> Tuple[bool, RobotState]:
        success = execute_instruction(instruction, state)
        new_state = update_robot_state(state, success)
        return success, new_state
    return Eff(run_step)
```

## 🔧 Running the Examples

### Prerequisites
```bash
# Install dependencies
uv pip install -e ".[models,dev]"

# Set environment variables
export OPENAI_API_KEY="sk-your-key-here"
export RAY_DISABLE_IMPORT_WARNING=1
```

### Command Line Execution
```bash
# From project root
cd retriever/planning/examples

# Run original example
python example_oracle.py

# Run typed example  
python example_oracle_typed.py

# Run Flow-based example (recommended)
python example_oracle_flow.py
```

### PyCharm Integration
Three run configurations are provided:
- **Original Oracle Example**
- **Typed Oracle Example** 
- **Flow Oracle Example**

Select any configuration from the Run menu and execute.

## 📚 Learning Path

**Recommended progression for understanding the architecture:**

1. **Start with `example_oracle.py`** - Understand the basic bilevel planning concepts
2. **Move to `example_oracle_typed.py`** - See how types improve reliability
3. **Study `example_oracle_flow.py`** - Learn the composable pipeline patterns

## 🎨 Extending the Examples

### Adding New Perception Modules
```python
class YourPerceptionModule:
    def __call__(self, image: RGBImage) -> YourDetectionType:
        # Your perception logic here
        return detections

# Integrate into pipeline
perception_flow = Flow.from_module(YourPerceptionModule())
pipeline = perception_flow.then(existing_pipeline)
```

### Creating Custom Planning Modules
```python
class YourPlannerModule:
    def __call__(self, task: TaskGoal) -> StructuredPlan:
        # Your planning logic here
        return plan

# Compose with other modules
planning_pipeline = (
    Flow.from_module(YourPlannerModule())
    .then(Flow.from_module(PlanValidator()))
    .then(Flow.from_module(PlanOptimizer()))
)
```

### Parallel Processing Examples
```python
# Multi-algorithm planning
multi_planner = (
    rrt_planner
    .fanout(prm_planner)
    .fanout(a_star_planner)
    .then(Flow.from_module(select_best_plan))
)

# Multi-sensor fusion
sensor_fusion = (
    camera_pipeline
    .fanout(lidar_pipeline)
    .fanout(imu_pipeline)
    .then(Flow.from_module(fuse_sensor_data))
)
```

## 🏆 Best Practices

1. **Use Flow composition** for production robotics systems
2. **Implement Module[I, O]** for all pipeline components
3. **Leverage fanout()** for parallel processing and sensor fusion
4. **Use Eff monad** for stateful robot operations
5. **Add comprehensive type annotations** for better debugging
6. **Test modules independently** before composing into pipelines
7. **Handle errors gracefully** with structured ExecutionStatus

## 🔗 Related Documentation

- [Core Types Documentation](../../core/types.py) - Understanding the type system
- [Flow Framework Documentation](../../core/flow.py) - Pipeline composition patterns
- [Skills Documentation](../../skills/) - Robot skill implementations
- [Models Documentation](../../models/) - Foundation model integration

---

**💡 Pro Tip**: The Flow-based example (`example_oracle_flow.py`) represents the recommended architecture for new Retriever projects. It provides the best balance of type safety, composability, and maintainability for production robotics systems.