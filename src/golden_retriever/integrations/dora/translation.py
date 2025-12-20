"""
Flow to Dora Translation Layer - Complete Implementation

This module provides the complete translation from Flow definitions to working
Dora operators, bridging the gap between Flow.run() methods and executable
Dora Python operators.

Key Components:
- FlowInstanceSerializer: Embeds Flow instances in generated operators
- WorkingOperatorGenerator: Creates operators that execute Flow.run() methods  
- PipelineTranslator: Converts Flow pipelines to multi-node Dora dataflows
"""

import inspect
import os
import tempfile
import time
import ast
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Set
from dataclasses import dataclass

from ...core.flow import Flow
from ...core.types import Pipeline
from ...types.registry import get_global_registry, get_registered_types
from .serialization import ArrowMessageSerializer


@dataclass
class DoraConfig:
    """Configuration for generated Dora dataflow."""
    yaml_path: str
    operators: List[str]
    output_dir: str


class TypeDependencyAnalyzer:
    """
    Analyzes Flow class dependencies and generates smart imports using type registry.
    
    This class replaces manual type imports with automatic dependency detection
    and registry-based import generation.
    """
    
    def __init__(self):
        self.registry = get_global_registry()
        self.registered_types = get_registered_types()
    
    def analyze_flow_dependencies(self, flow: Flow) -> Dict[str, Any]:
        """
        Analyze Flow class to extract all type dependencies.
        
        Args:
            flow: The Flow instance to analyze
            
        Returns:
            Dict containing imports, type hints, and other dependencies
        """
        flow_class = flow.__class__
        
        # Extract class source code
        try:
            class_source = inspect.getsource(flow_class)
        except Exception:
            class_source = ""
        
        # Extract method sources
        init_source = self._extract_method_source(flow, '__init__')
        run_source = self._extract_method_source(flow, 'run')
        
        # Analyze all sources for dependencies
        all_sources = [class_source, init_source, run_source]
        dependencies = self._analyze_sources_for_dependencies(all_sources)
        
        return {
            "imports": dependencies["imports"],
            "type_hints": dependencies["type_hints"],
            "external_modules": dependencies["external_modules"],
            "registered_types": dependencies["registered_types"],
            "unresolved_names": dependencies["unresolved_names"]
        }
    
    def _extract_method_source(self, flow: Flow, method_name: str) -> str:
        """Extract source code for a specific method."""
        try:
            method = getattr(flow, method_name, None)
            if method:
                return inspect.getsource(method)
        except Exception:
            pass
        return ""
    
    def _analyze_sources_for_dependencies(self, sources: List[str]) -> Dict[str, Any]:
        """Analyze source code to find all dependencies."""
        imports = set()
        type_hints = set()
        external_modules = set()
        registered_types = set()
        unresolved_names = set()
        
        # Parse each source with AST
        for source in sources:
            if not source.strip():
                continue
                
            try:
                tree = ast.parse(source)
                visitor = DependencyVisitor(self.registered_types)
                visitor.visit(tree)
                
                imports.update(visitor.imports)
                type_hints.update(visitor.type_hints)
                external_modules.update(visitor.external_modules)
                registered_types.update(visitor.registered_types)
                unresolved_names.update(visitor.unresolved_names)
                
            except Exception as e:
                # Fallback to regex analysis
                fallback_deps = self._regex_dependency_analysis(source)
                imports.update(fallback_deps["imports"])
                type_hints.update(fallback_deps["type_hints"])
                unresolved_names.update(fallback_deps["unresolved_names"])
        
        return {
            "imports": list(imports),
            "type_hints": list(type_hints), 
            "external_modules": list(external_modules),
            "registered_types": list(registered_types),
            "unresolved_names": list(unresolved_names)
        }
    
    def _regex_dependency_analysis(self, source: str) -> Dict[str, List[str]]:
        """Fallback regex-based dependency analysis."""
        imports = []
        type_hints = []
        unresolved_names = []
        
        # Find type annotations
        type_pattern = r':\s*([A-Z][A-Za-z0-9_]*(?:\[[^\]]+\])?)'
        for match in re.finditer(type_pattern, source):
            type_name = match.group(1)
            # Extract base type name (before brackets)
            base_type = re.match(r'([A-Z][A-Za-z0-9_]*)', type_name)
            if base_type:
                type_hints.append(base_type.group(1))
        
        # Find direct class usage (like RGBImage(...))
        class_usage_pattern = r'\b([A-Z][A-Za-z0-9_]*)\s*\('
        for match in re.finditer(class_usage_pattern, source):
            class_name = match.group(1)
            if class_name not in ['List', 'Dict', 'Optional', 'Union', 'Tuple']:  # Skip generic types
                unresolved_names.append(class_name)
        
        return {
            "imports": imports,
            "type_hints": type_hints,
            "unresolved_names": unresolved_names
        }
    
    def generate_smart_imports(self, dependencies: Dict[str, Any]) -> str:
        """
        Generate smart import statements using type registry.
        
        Args:
            dependencies: Dependency analysis results
            
        Returns:
            Python import statements as string
        """
        import_lines = []
        
        # Group imports by module
        module_imports = {}
        
        # Process registered types
        for type_name in dependencies["registered_types"]:
            type_info = self.registry.get_type_info(type_name)
            if type_info:
                module = type_info.module
                if module not in module_imports:
                    module_imports[module] = []
                module_imports[module].append(type_name)
        
        # Process unresolved names by checking registry
        for name in dependencies["unresolved_names"]:
            type_info = self.registry.get_type_info(name)
            if type_info:
                module = type_info.module
                if module not in module_imports:
                    module_imports[module] = []
                if name not in module_imports[module]:
                    module_imports[module].append(name)
            else:
                # Check common modules
                if name in ['cv2']:
                    import_lines.append(f"import {name}")
                elif name in ['RGBImage', 'Detection', 'BoundingBox']:
                    # These are likely our core types
                    if 'retriever.types.core_types' not in module_imports:
                        module_imports['retriever.types.core_types'] = []
                    module_imports['retriever.types.core_types'].append(name)
                elif name in ['CameraFlow', 'ColorDetector', 'OpenCVVisualizer']:
                    # These are vision components
                    if 'retriever.flows.vision.visualization' not in module_imports:
                        module_imports['retriever.flows.vision.visualization'] = []
                    module_imports['retriever.flows.vision.visualization'].append(name)
        
        # Always add cv2 import since it's commonly used in vision flows
        if 'cv2' not in [line.replace('import ', '') for line in import_lines if line.startswith('import ')]:
            import_lines.append("import cv2")
        
        # Generate from imports
        for module, names in sorted(module_imports.items()):
            if names:
                import_lines.append(f"from {module} import {', '.join(sorted(names))}")
        
        # Add standard library imports
        std_imports = []
        if any('List' in hint for hint in dependencies["type_hints"]):
            std_imports.append("List")
        if any('Dict' in hint for hint in dependencies["type_hints"]):
            std_imports.append("Dict")
        if any('Optional' in hint for hint in dependencies["type_hints"]):
            std_imports.append("Optional")
        
        if std_imports:
            import_lines.insert(0, f"from typing import {', '.join(sorted(std_imports))}")
        
        return '\n'.join(import_lines)


class DependencyVisitor(ast.NodeVisitor):
    """AST visitor to extract dependencies from Python source code."""
    
    def __init__(self, registered_types: Dict[str, Any]):
        self.registered_types = registered_types
        self.imports = set()
        self.type_hints = set()
        self.external_modules = set()
        self.registered_types_found = set()
        self.unresolved_names = set()
    
    def visit_Import(self, node):
        """Handle 'import module' statements."""
        for alias in node.names:
            self.imports.add(alias.name)
            self.external_modules.add(alias.name.split('.')[0])
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Handle 'from module import name' statements."""
        if node.module:
            self.external_modules.add(node.module.split('.')[0])
            for alias in node.names:
                self.imports.add(f"{node.module}.{alias.name}")
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Handle name references (variables, classes, functions)."""
        name = node.id
        
        # Check if it's a registered type
        if name in self.registered_types:
            self.registered_types_found.add(name)
        # Check for common type patterns
        elif name in ['RGBImage', 'Detection', 'BoundingBox', 'CameraFlow', 'ColorDetector', 'OpenCVVisualizer']:
            self.unresolved_names.add(name)
        # Check for type annotations
        elif name in ['List', 'Dict', 'Optional', 'Union', 'Tuple']:
            self.type_hints.add(name)
        
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Handle function/class calls."""
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name in self.registered_types:
                self.registered_types_found.add(name)
            elif name not in ['print', 'len', 'range', 'enumerate']:  # Skip builtins
                self.unresolved_names.add(name)
        
        self.generic_visit(node)


class FlowInstanceSerializer:
    """
    Serializes Flow instances for embedding in generated Dora operators.
    
    This extracts all necessary information from Flow instances to recreate
    them inside generated Dora operators.
    """
    
    def serialize_flow_instance(self, flow: Flow) -> Dict[str, Any]:
        """
        Extract all information needed to recreate Flow in operator.
        
        Args:
            flow: The Flow instance to serialize
            
        Returns:
            Dict containing class info, source code, dependencies
        """
        flow_class = flow.__class__
        
        return {
            "class_name": flow_class.__name__,
            "module_path": flow_class.__module__,
            "run_method_source": self._extract_run_method(flow),
            "init_method_source": self._extract_init_method(flow),
            "class_source": self._extract_class_source(flow),
            "rate_annotation": self._extract_rate_annotation(flow),
            "dependencies": self._extract_dependencies(flow),
            "instance_attributes": self._extract_instance_attributes(flow)
        }
    
    def _extract_run_method(self, flow: Flow) -> str:
        """Extract the run() method source code."""
        try:
            return inspect.getsource(flow.run)
        except Exception as e:
            # Fallback for dynamically defined methods
            return f'''
    def run(self, input_data):
        """Generated run method - original source not available: {e}"""
        # This is a placeholder - original Flow.run() method could not be extracted
        raise NotImplementedError("Flow.run() method source extraction failed")
'''
    
    def _extract_init_method(self, flow: Flow) -> str:
        """Extract the __init__ method source code."""
        try:
            return inspect.getsource(flow.__init__)
        except Exception:
            return '''
    def __init__(self):
        """Generated init method - original source not available"""
        super().__init__()
'''
    
    def _extract_class_source(self, flow: Flow) -> str:
        """Extract the complete class source code."""
        try:
            return inspect.getsource(flow.__class__)
        except Exception as e:
            # Fallback: create minimal class definition
            class_name = flow.__class__.__name__
            return f'''
class {class_name}(Flow):
    """Generated class definition - original source not available: {e}"""
    
    def __init__(self):
        super().__init__()
    
    def run(self, input_data):
        raise NotImplementedError("Original run method not available")
'''
    
    def _extract_rate_annotation(self, flow: Flow) -> Optional[str]:
        """Extract @flow(rate="...") annotation if present."""
        # Check for rate attribute set by @flow decorator
        return (
            getattr(flow, '_flow_rate', None)
            or getattr(flow.__class__, '_flow_rate', None)
            or getattr(flow, '_rate', None)
            or getattr(flow.__class__, '_rate', None)
        )
    
    def _extract_dependencies(self, flow: Flow) -> List[str]:
        """Extract import dependencies from Flow class."""
        # Get the module where Flow is defined
        module = inspect.getmodule(flow.__class__)
        if not module:
            return []
        
        # Extract imports (simplified approach)
        dependencies = []
        try:
            source = inspect.getsource(module)
            lines = source.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    dependencies.append(line)
        except Exception:
            pass
        
        return dependencies
    
    def _extract_instance_attributes(self, flow: Flow) -> Dict[str, Any]:
        """Extract instance attributes that need to be preserved."""
        # Get all attributes that aren't methods or special attributes
        attributes = {}
        for name, value in flow.__dict__.items():
            if not name.startswith('_') and not callable(value):
                # Try to serialize the value
                try:
                    # For now, only handle simple types
                    if isinstance(value, (str, int, float, bool, list, dict)):
                        attributes[name] = value
                except Exception:
                    pass
        
        return attributes


class WorkingOperatorGenerator:
    """
    Generates working Dora operators that execute Flow.run() methods.
    
    This replaces the placeholder operator generation with real implementation
    that embeds and executes actual Flow instances.
    """
    
    def __init__(self):
        self.serializer = FlowInstanceSerializer()
        self.dependency_analyzer = TypeDependencyAnalyzer()
    
    def generate_working_operator(self, flow: Flow, flow_name: str) -> str:
        """
        Generate a working Dora operator that executes the given Flow.
        
        Args:
            flow: The Flow instance to execute
            flow_name: Name for the generated operator
            
        Returns:
            Complete Python source code for working Dora operator
        """
        flow_data = self.serializer.serialize_flow_instance(flow)
        
        # Analyze dependencies for smart imports
        dependencies = self.dependency_analyzer.analyze_flow_dependencies(flow)
        smart_imports = self.dependency_analyzer.generate_smart_imports(dependencies)
        
        # Generate the operator code
        operator_code = f'''#!/usr/bin/env python3
"""
Generated Dora Operator for {flow_data["class_name"]}
Auto-generated on {datetime.now().isoformat()}

This operator executes the actual Flow.run() method in Dora runtime.
"""

import sys
import os
import time
import traceback
from typing import Any, Optional

# Ensure retriever imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
retriever_root = os.path.join(current_dir, '../../..')
if retriever_root not in sys.path:
    sys.path.insert(0, retriever_root)

try:
    from dora import Node
    DORA_AVAILABLE = True
except ImportError:
    print("Warning: Dora not available, operator will fail")
    DORA_AVAILABLE = False

# Import serialization
try:
    from retriever.integrations.dora.serialization import ArrowMessageSerializer
    from retriever.core.flow import Flow
    {self._generate_flow_import(flow_data)}
    RETRIEVER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Retriever imports failed: {{e}}")
    RETRIEVER_AVAILABLE = False

# Smart imports generated from type registry
{smart_imports}

# Additional imports for Dora operators
import numpy as np

# Flow class from: {flow_data["module_path"]}.{flow_data["class_name"]}
{self._generate_flow_class_definition(flow_data)}

class Operator:
    """Dora operator that executes {flow_data["class_name"]}.run() method."""
    
    def __init__(self):
        if not DORA_AVAILABLE:
            raise RuntimeError("Dora not available")
        if not RETRIEVER_AVAILABLE:
            raise RuntimeError("Retriever not available")
            
        # Initialize serialization
        self.serializer = ArrowMessageSerializer()
        
        # Create the actual Flow instance
        try:
            self.flow = {flow_data["class_name"]}()
            {self._generate_attribute_assignments(flow_data["instance_attributes"])}
            print(f"✅ Initialized {{self.flow.__class__.__name__}} in Dora operator")
        except Exception as e:
            print(f"❌ Failed to initialize Flow: {{e}}")
            raise
        
        # FRP rate coordination
        self.target_rate = {self._extract_rate_millis(flow_data["rate_annotation"])}
        self.last_execution_time = 0.0
        self.execution_count = 0
        
    def on_event(self, dora_event, send_output):
        """Dora operator event handler - matches Dora Python API."""
        try:
            from dora import DoraStatus
        except ImportError:
            print("⚠️  DoraStatus not available - using fallback")
            from enum import Enum
            class DoraStatus(Enum):
                CONTINUE = 0
                STOP = 1
        
        try:
            event_type = dora_event.get("type", "")
            
            if event_type == "INPUT" or "tick" in dora_event or "input" in dora_event:
                current_time = time.time()
                
                # Handle rate coordination
                if self._should_execute(current_time):
                    # Deserialize input data using Arrow
                    input_data = self._deserialize_input(dora_event)
                    
                    # Execute the actual Flow.run() method
                    execution_start = time.time()
                    result = self.flow.run(input_data)
                    execution_time = (time.time() - execution_start) * 1000
                    
                    # Debug: Print execution details
                    if self.execution_count % 10 == 0:  # Every 10th execution
                        print(f"🔍 Flow {{self.flow.__class__.__name__}} executed:")
                        print(f"   Input type: {{type(input_data)}}")
                        print(f"   Result type: {{type(result)}}")
                        if hasattr(result, '__len__'):
                            print(f"   Result length: {{len(result)}}")
                            if hasattr(result, '__iter__') and len(result) > 0:
                                first_item = next(iter(result))
                                print(f"   First item type: {{type(first_item)}}")
                                if hasattr(first_item, 'label'):
                                    print(f"   First detection: {{first_item.label}} ({{first_item.confidence:.2f}})")
                    
                    # Serialize result using PyArrow (Dora standard)
                    output_data, output_metadata = self._serialize_output(result)
                    
                    # Send to Dora using standard format
                    send_output("output", output_data, output_metadata)
                    
                    # Update timing
                    self.last_execution_time = current_time
                    self.execution_count += 1
                    
                    # Periodic status
                    if self.execution_count % 30 == 0:
                        print(f"🔥 Executed {{self.execution_count}} cycles, last took {{execution_time:.1f}}ms")
                
                return DoraStatus.CONTINUE
                
            elif event_type == "STOP":
                print(f"🛑 Stopping operator after {{self.execution_count}} executions")
                self._cleanup()
                return DoraStatus.STOP
                
            return DoraStatus.CONTINUE
                
        except Exception as e:
            print(f"❌ Flow execution error: {{e}}")
            print(f"Traceback: {{traceback.format_exc()}}")
            return DoraStatus.CONTINUE  # Continue despite errors
    
    def _should_execute(self, current_time: float) -> bool:
        """Check if enough time has passed for next execution."""
        if self.target_rate <= 0:
            return True  # No rate limiting
            
        time_since_last = (current_time - self.last_execution_time) * 1000
        return time_since_last >= self.target_rate
    
    def _deserialize_input(self, event: dict) -> Any:
        """Deserialize input from Dora event."""
        try:
            # Handle different input formats
            if "value" in event:
                # Check if it's pickle data (bytes)
                try:
                    import pickle
                    data = pickle.loads(event["value"])
                    
                    # Check if it's our special format with to_arrow/from_arrow
                    if isinstance(data, dict) and 'type' in data and 'data' in data:
                        obj_type = data['type']
                        arrow_data = data['data']
                        
                        # Import the type and reconstruct using from_arrow
                        if obj_type == 'RGBImage':
                            from retriever.types.core_types import RGBImage
                            return RGBImage.from_arrow(arrow_data)
                        elif obj_type == 'Detection':
                            from retriever.types.core_types import Detection
                            return Detection.from_arrow(arrow_data)
                        else:
                            return data  # Fallback to original data
                    else:
                        return data  # Direct pickle data
                        
                except Exception:
                    # Raw Arrow data - return as is
                    return event["value"]
            elif "data" in event:
                # Raw bytes - try pickle first
                try:
                    import pickle
                    return pickle.loads(event["data"])
                except Exception:
                    return event["data"]
            else:
                # Timer tick or no data
                return None
                
        except Exception as e:
            print(f"⚠️  Input deserialization failed: {{e}}, using raw data")
            return event.get("value") or event.get("data")
    
    def _serialize_output(self, result: Any) -> Any:
        """Serialize output for Dora using PyArrow arrays."""
        try:
            # Handle RGBImage: flatten the numpy array and send with metadata
            if hasattr(result, 'data') and hasattr(result.data, 'shape'):
                import pyarrow as pa
                # Flatten image data and store shape info
                flat_data = result.data.ravel()
                metadata = {{
                    'type': result.__class__.__name__,
                    'shape': result.data.shape,
                    'timestamp': getattr(result, 'timestamp', None),
                    'camera_id': getattr(result, 'camera_id', 'default')
                }}
                return pa.array(flat_data), metadata
                
            # Handle list of detections
            elif isinstance(result, list) and result and hasattr(result[0], 'label'):
                import pyarrow as pa
                # Serialize detections as structured data
                detection_data = []
                for det in result:
                    detection_data.append({{
                        'label': det.label,
                        'confidence': det.confidence,
                        'bbox_x': det.bbox.x,
                        'bbox_y': det.bbox.y,
                        'bbox_width': det.bbox.width,
                        'bbox_height': det.bbox.height
                    }})
                return pa.array([str(detection_data)]), {{'type': 'DetectionList', 'count': len(result)}}
                
            # Fallback: convert to arrow array  
            else:
                import pyarrow as pa
                import pickle
                return pa.array([pickle.dumps(result)]), {{'type': 'pickle'}}
                
        except Exception as e:
            print(f"⚠️  Serialization failed: {{e}}, using pickle fallback")
            import pyarrow as pa
            import pickle
            return pa.array([pickle.dumps(result)]), {{'type': 'pickle'}}
    
    def _infer_data_type(self, data: Any) -> str:
        """Infer data type for optimized serialization."""
        import numpy as np
        
        if isinstance(data, np.ndarray):
            if len(data.shape) == 3 and data.shape[2] in [1, 3, 4]:
                return "image"
            elif len(data.shape) == 2 and data.shape[1] >= 3:
                return "point_cloud"
            else:
                return "numpy_array"
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            if "box" in data[0] or "bbox" in data[0]:
                return "detections"
        elif isinstance(data, dict):
            return "dict"
        elif isinstance(data, list):
            return "list"
        else:
            return "generic"
    
    def _cleanup(self):
        """Clean up Flow resources."""
        try:
            if hasattr(self.flow, 'cleanup'):
                self.flow.cleanup()
                print("✅ Flow cleanup completed")
        except Exception as e:
            print(f"⚠️  Flow cleanup failed: {{e}}")


# Dora operator instance - instantiated when module is imported
operator = Operator()

def on_event(dora_event, send_output):
    """Entry point called by Dora runtime."""
    operator.on_event(dora_event, send_output)
'''
        return operator_code
    
    def _format_dependencies(self, dependencies: List[str]) -> str:
        """Format import dependencies for the operator."""
        if not dependencies:
            return "# No additional dependencies"
        
        formatted = []
        for dep in dependencies:
            # Skip retriever imports as they're handled separately
            if 'retriever' not in dep:
                formatted.append(f"# {dep}")  # Comment out for safety
        
        return '\n'.join(formatted) if formatted else "# No additional dependencies"
    
    def _generate_attribute_assignments(self, attributes: Dict[str, Any]) -> str:
        """Generate code to set instance attributes."""
        if not attributes:
            return ""
        
        assignments = []
        for name, value in attributes.items():
            assignments.append(f"            self.flow.{name} = {repr(value)}")
        
        return '\n' + '\n'.join(assignments) if assignments else ""
    
    def _generate_flow_import(self, flow_data: Dict[str, Any]) -> str:
        """Generate import statement, avoiding duplicates and handling script-defined flows."""
        module_path = flow_data["module_path"]
        class_name = flow_data["class_name"]
        
        # Avoid duplicate import if it's the base Flow class
        if module_path == "retriever.core.flow" and class_name == "Flow":
            return "# Flow already imported above"
        
        # If this is from __main__ (script), embed the class definition instead of importing
        if module_path == "__main__":
            return f"# {class_name} will be embedded below (from script)"
        
        return f"from {module_path} import {class_name}"
    
    def _generate_flow_class_definition(self, flow_data: Dict[str, Any]) -> str:
        """Generate Flow class definition for embedding in operator."""
        module_path = flow_data["module_path"]
        class_name = flow_data["class_name"]
        
        # If this is from __main__ (script), embed the full class definition
        if module_path == "__main__":
            class_source = flow_data["class_source"]
            # Remove @flow decorator since Dora handles rate via YAML
            clean_source = self._remove_flow_decorator(class_source)
            return f"""# Embedded Flow class from script (decorator removed for Dora)
{clean_source}"""
        else:
            # For proper modules, just add a comment (class is imported)
            return f"# {class_name} imported from {module_path}"
    
    def _remove_flow_decorator(self, class_source: str) -> str:
        """Remove @flow decorator from class source since Dora handles timing."""
        import re
        # Remove @flow decorator lines (including multi-line decorators)
        lines = class_source.split('\n')
        clean_lines = []
        skip_decorator = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('@flow'):
                skip_decorator = True
                continue
            elif skip_decorator and (stripped.startswith('class ') or (stripped and not stripped.startswith('@'))):
                skip_decorator = False
                clean_lines.append(line)
            elif not skip_decorator:
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def _extract_rate_millis(self, rate_annotation: Optional[str]) -> int:
        """Convert rate annotation to milliseconds."""
        if not rate_annotation:
            return 50  # Default 50ms = 20Hz
        
        rate = rate_annotation.lower()
        if rate.endswith('hz'):
            hz = float(rate[:-2])
            return int(1000 / hz)
        elif rate.endswith('ms'):
            return int(float(rate[:-2]))
        elif rate.endswith('s'):
            return int(float(rate[:-1]) * 1000)
        else:
            # Try to parse as number (assume Hz)
            try:
                hz = float(rate)
                return int(1000 / hz)
            except ValueError:
                return 50  # Default


class PipelineTranslator:
    """
    Translates complete Flow pipelines to multi-node Dora dataflows.
    
    This is the main interface for converting Flow compositions like
    'camera >> detector' into working Dora configurations.
    """
    
    def __init__(self):
        self.operator_generator = WorkingOperatorGenerator()
    
    def translate_pipeline(self, pipeline: Pipeline, output_dir: str) -> DoraConfig:
        """
        Complete translation from Pipeline to working Dora execution.
        
        Args:
            pipeline: The Pipeline to translate
            output_dir: Directory for generated files
            
        Returns:
            DoraConfig with paths to generated files
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract flows from pipeline
        flows = self._extract_flows_from_pipeline(pipeline)
        trigger_edges = self._collect_trigger_edges_from_pipeline(pipeline)
        multiinput_edges = self._collect_multiinput_edges_from_pipeline(pipeline)
        
        print(f"🔍 Extracted {len(flows)} flows from pipeline:")
        for i, flow in enumerate(flows):
            print(f"   Flow {i+1}: {flow.__class__.__name__}")
        
        if not flows:
            raise ValueError("No flows found in pipeline")
        
        # Generate operators for each flow (use derived node ids)
        operator_paths = []
        for i, flow in enumerate(flows):
            flow_name = self._derive_node_id(flow, i)
            operator_code = self.operator_generator.generate_working_operator(flow, flow_name)
            
            operator_path = os.path.join(output_dir, f"{flow_name}_op.py")
            with open(operator_path, 'w') as f:
                f.write(operator_code)
            
            operator_paths.append(operator_path)
            print(f"✅ Generated operator: {operator_path}")
        
        # Generate dataflow YAML
        yaml_content = self._generate_dataflow_yaml(flows, trigger_edges, multiinput_edges)
        yaml_path = os.path.join(output_dir, "dataflow.yml")
        
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"✅ Generated dataflow: {yaml_path}")
        
        return DoraConfig(
            yaml_path=yaml_path,
            operators=operator_paths,
            output_dir=output_dir
        )
    
    def _extract_flows_from_pipeline(self, pipeline: Pipeline) -> List[Flow]:
        """Extract individual Flow instances from Pipeline composition."""
        # For now, handle simple cases - this can be enhanced for complex compositions
        flows = []
        
        if hasattr(pipeline, 'composition_tree') and pipeline.composition_tree:
            # Navigate the composition tree to extract flows
            flows = self._extract_from_composition_tree(pipeline.composition_tree)
        elif hasattr(pipeline, '_flows'):
            # Direct access to flows list
            flows = pipeline._flows
        else:
            # Try to get flows from get_flows method
            try:
                flows = pipeline.get_flows()
                # Check if any of these flows are compositions that need decomposing
                decomposed_flows = []
                for flow in flows:
                    if hasattr(flow, '_node') and flow._node:
                        # This flow is a composition - decompose it
                        decomposed_flows.extend(self._extract_flows_from_node(flow._node))
                    else:
                        # Regular flow - keep as is
                        decomposed_flows.append(flow)
                flows = decomposed_flows
            except Exception:
                # Last resort: assume pipeline wraps a single flow
                if hasattr(pipeline, 'flow'):
                    single_flow = pipeline.flow
                    # Check if this is actually a composition that needs decomposing
                    if hasattr(single_flow, '_node') and single_flow._node:
                        flows = self._extract_flows_from_node(single_flow._node)
                    else:
                        flows = [single_flow]
                else:
                    raise ValueError(f"Cannot extract flows from pipeline: {type(pipeline)}")
        
        return flows
    
    def _collect_trigger_edges_from_pipeline(self, pipeline: Pipeline) -> List[Tuple[str, str]]:
        """Collect trigger relationships (trigger_flow -> target_flow)."""
        edges: set[Tuple[str, str]] = set()
        try:
            tree = getattr(pipeline, 'composition_tree', None)
            if tree and getattr(tree, 'flow', None) is not None:
                flow = tree.flow
                if hasattr(flow, '_node') and flow._node:
                    self._collect_trigger_edges_from_node(flow._node, edges)
        except Exception:
            pass
        return list(edges)
    
    def _collect_trigger_edges_from_node(self, node, edges: set[Tuple[str, str]]):
        from ...core.flow import ModuleNode, ThenNode, FanoutNode, TriggeredNode, MultiInputNode
        if isinstance(node, TriggeredNode):
            src = node.trigger_flow.__class__.__name__.lower()
            dst = node.target_flow.__class__.__name__.lower()
            edges.add((src, dst))
            # Recurse into children
            self._collect_trigger_edges_from_node(node.trigger_flow._node, edges) if hasattr(node.trigger_flow, '_node') else None
            self._collect_trigger_edges_from_node(node.target_flow._node, edges) if hasattr(node.target_flow, '_node') else None
        elif isinstance(node, ThenNode):
            self._collect_trigger_edges_from_node(node.first._node, edges)
            self._collect_trigger_edges_from_node(node.second._node, edges)
        elif isinstance(node, FanoutNode):
            self._collect_trigger_edges_from_node(node.first._node, edges)
            self._collect_trigger_edges_from_node(node.second._node, edges)
        elif isinstance(node, MultiInputNode):
            for f in node.input_flows:
                self._collect_trigger_edges_from_node(f._node, edges)
            self._collect_trigger_edges_from_node(node.coordinator_flow._node, edges)
        elif isinstance(node, ModuleNode):
            return
    
    def _extract_flows_from_node(self, node) -> List[Flow]:
        """Extract individual flows from Flow composition nodes."""
        flows = []
        
        # Import here to avoid circular imports
        from ...core.flow import ModuleNode, ThenNode, FanoutNode, TriggeredNode, MultiInputNode
        
        if isinstance(node, ModuleNode):
            # This is a leaf node - try to find the original flow
            # For now, we can't easily recover the original Flow instance from ModuleNode
            # So we create a temporary flow wrapper
            flows.append(self._create_flow_from_module_node(node))
            
        elif isinstance(node, ThenNode):
            # Sequential composition: extract both flows
            # ThenNode contains the original Flow objects directly!
            flows.append(node.first)  # Original first flow
            flows.append(node.second)  # Original second flow
            
        elif isinstance(node, FanoutNode):  
            # Parallel composition: extract both flows
            # FanoutNode also contains the original Flow objects directly!
            flows.append(node.first)   # Original first flow
            flows.append(node.second)  # Original second flow
            
        elif isinstance(node, TriggeredNode):
            # Event-driven composition: extract trigger flow and target flow as separate nodes
            flows.append(node.trigger_flow)  # Monitoring flow
            flows.append(node.target_flow)   # Response flow
            
        elif isinstance(node, MultiInputNode):
            # Multi-input coordination: extract all input flows + coordinator as separate nodes
            flows.extend(node.input_flows)   # All input flows  
            flows.append(node.coordinator_flow)  # Coordinator flow
        
        return flows
    
    def _create_flow_from_module_node(self, node) -> Flow:
        """Create a Flow wrapper around a ModuleNode function."""
        # This is a workaround - we create a simple Flow that wraps the function
        from ...core.flow import Flow
        
        class GeneratedFlow(Flow):
            def __init__(self, func):
                super().__init__()
                self.func = func
                
            def run(self, input_data):
                return self.func(input_data)
        
        return GeneratedFlow(node.func)
    
    def _extract_from_composition_tree(self, tree) -> List[Flow]:
        """Extract flows from composition tree structure."""
        flows = []
        
        if hasattr(tree, 'flow') and tree.flow:
            # Leaf node with actual flow
            flow = tree.flow
            if hasattr(flow, '_node') and flow._node:
                # This flow is a composition - decompose it
                flows.extend(self._extract_flows_from_node(flow._node))
            else:
                flows.append(flow)
        elif hasattr(tree, 'first') and hasattr(tree, 'second'):
            # Composition node - recursively extract
            flows.extend(self._extract_from_composition_tree(tree.first))
            flows.extend(self._extract_from_composition_tree(tree.second))
        
        return flows
    
    def _collect_multiinput_edges_from_pipeline(self, pipeline: Pipeline) -> Dict[str, List[str]]:
        """Collect MultiInput relationships mapping coordinator -> list of source flows."""
        mapping: Dict[str, List[str]] = {}
        try:
            tree = getattr(pipeline, 'composition_tree', None)
            if tree and getattr(tree, 'flow', None) is not None:
                flow = tree.flow
                if hasattr(flow, '_node') and flow._node:
                    self._collect_multiinput_edges_from_node(flow._node, mapping)
        except Exception:
            pass
        return mapping

    def _collect_multiinput_edges_from_node(self, node, mapping: Dict[str, List[str]]):
        from ...core.flow import ModuleNode, ThenNode, FanoutNode, TriggeredNode, MultiInputNode
        if isinstance(node, MultiInputNode):
            dst = node.coordinator_flow.__class__.__name__.lower()
            sources = []
            for f in node.input_flows:
                sources.append(f.__class__.__name__.lower())
            mapping[dst] = sources
            # Recurse into inputs and coordinator
            for f in node.input_flows:
                if hasattr(f, '_node'):
                    self._collect_multiinput_edges_from_node(f._node, mapping)
            if hasattr(node.coordinator_flow, '_node'):
                self._collect_multiinput_edges_from_node(node.coordinator_flow._node, mapping)
        elif isinstance(node, ThenNode):
            self._collect_multiinput_edges_from_node(node.first._node, mapping)
            self._collect_multiinput_edges_from_node(node.second._node, mapping)
        elif isinstance(node, FanoutNode):
            self._collect_multiinput_edges_from_node(node.first._node, mapping)
            self._collect_multiinput_edges_from_node(node.second._node, mapping)
        elif isinstance(node, TriggeredNode):
            if hasattr(node.trigger_flow, '_node'):
                self._collect_multiinput_edges_from_node(node.trigger_flow._node, mapping)
            if hasattr(node.target_flow, '_node'):
                self._collect_multiinput_edges_from_node(node.target_flow._node, mapping)
        elif isinstance(node, ModuleNode):
            return

    def _generate_dataflow_yaml(self, flows: List[Flow], trigger_edges: List[Tuple[str, str]], multiinput_edges: Dict[str, List[str]]) -> str:
        """Generate Dora dataflow YAML for multiple flows."""
        yaml_lines = [
            "# Generated Dora dataflow with working Flow operators",
            f"# Generated on {datetime.now().isoformat()}",
            "",
            "nodes:"
        ]
        
        # Index trigger edges for quick lookup by destination (support multiple triggers)
        triggers_by_dst: Dict[str, List[str]] = {}
        for src, dst in trigger_edges:
            triggers_by_dst.setdefault(self._snake(dst), []).append(self._snake(src))

        # Build best-effort feedback edges from annotations: source -> target
        flow_names = [self._derive_node_id(f, i) for i, f in enumerate(flows)]
        feedback_edges: List[Tuple[str, str]] = []
        for f in flows:
            src = self._derive_node_id(f, flows.index(f))
            # Extract annotation
            fb_to = getattr(f, '_flow_feedback_to', None) or getattr(f.__class__, '_flow_feedback_to', None) \
                    or getattr(f, '_feedback_to', None) or getattr(f.__class__, '_feedback_to', None)
            if isinstance(fb_to, str):
                target = self._snake(fb_to.strip())
                # Map to known node names if present; otherwise keep the literal target
                if target in flow_names:
                    feedback_edges.append((src, target))
        
        # Index feedback edges by destination
        feedback_by_dst: Dict[str, List[str]] = {}
        for src, dst in feedback_edges:
            feedback_by_dst.setdefault(dst, []).append(src)
        # Track which nodes are feedback sources
        feedback_sources = {src for src, _ in feedback_edges}

        for i, flow in enumerate(flows):
            flow_name = self._derive_node_id(flow, i)
            operator_file = f"{flow_name}_op.py"
            rate = self._extract_flow_rate(flow)
            
            yaml_lines.extend([
                f"  - id: {flow_name}",
                f"    operator:",
                f"      python: {operator_file}",
                f"      inputs:"
            ])
            
            if i == 0 and flow_name not in multiinput_edges:
                # First flow gets timer input
                timer_interval = self._convert_rate_to_millis(rate)
                yaml_lines.append(f"        tick: dora/timer/millis/{timer_interval}")
            elif flow_name not in multiinput_edges:
                # Subsequent flows get data from previous flow
                prev_flow_name = flows[i-1].__class__.__name__.lower()
                yaml_lines.append(f"        input: {prev_flow_name}/output")

            # Add trigger inputs if present (support multiple triggers as trigger, trigger_2, ...)
            if flow_name in triggers_by_dst:
                sources = triggers_by_dst[flow_name]
                for idx, src in enumerate(sources):
                    key = "trigger" if idx == 0 else f"trigger_{idx+1}"
                    yaml_lines.append(f"        {key}: {src}/output")

            # Add multi-input edges: additional input_N ports for coordinator nodes
            if flow_name in multiinput_edges:
                sources = multiinput_edges[flow_name]
                # Prefer readable, deterministic input names based on source node IDs
                # Fallback to input_1, input_2 if name conflicts arise.
                used = set()
                for idx, src in enumerate(sources):
                    base_key = self._snake(src)  # use producer node id as input key
                    key = base_key if base_key not in used else f"input_{idx+1}"
                    used.add(key)
                    yaml_lines.append(f"        {key}: {src}/output")
            
            yaml_lines.append(f"      outputs:")
            yaml_lines.append(f"        - output")
            # Add feedback output if this node is a feedback source
            if flow_name in feedback_sources:
                yaml_lines.append(f"        - feedback")
            yaml_lines.append("")

            # If this node expects feedback inputs (is a target), append them to inputs
            if flow_name in feedback_by_dst:
                sources = feedback_by_dst[flow_name]
                for idx, src in enumerate(sources):
                    key = "feedback" if idx == 0 else f"feedback_{idx+1}"
                    yaml_lines.append(f"        {key}: {src}/feedback")
        
        # Add feedback inputs for targets (after nodes are listed)
        # Dora YAML may also allow embedding directly in node definitions; here we keep a simple structure
        # by having inputs defined above. If multiple feedback sources target same node, enumerate suffixes.
        # Note: This section is kept minimal; future pass can restructure for clarity if needed.
        # (Already handled above when defining inputs.)
        
        return '\n'.join(yaml_lines)

    def _derive_node_id(self, flow: Flow, idx: int | None = None) -> str:
        """Derive a stable, human-readable node id for a Flow.

        Prefers concrete subclass names; falls back to names from underlying nodes
        such as MultiInputNode.coordinator_flow or TriggeredNode.target_flow.
        """
        try:
            # Prefer concrete subclass names
            if flow.__class__.__name__ != 'Flow':
                return self._snake(flow.__class__.__name__)
            # Inspect underlying node
            node = getattr(flow, '_node', None)
            if node is not None:
                from ...core.flow import MultiInputNode, TriggeredNode
                if isinstance(node, MultiInputNode):
                    return self._snake(node.coordinator_flow.__class__.__name__)
                if isinstance(node, TriggeredNode):
                    return self._snake(node.target_flow.__class__.__name__)
        except Exception:
            pass
        # Fallback to generic name with index suffix to avoid collisions
        suffix = str(idx) if idx is not None else str(id(flow) % 10000)
        return f"flow_{suffix}"

    def _snake(self, name: str) -> str:
        """Convert CamelCase or mixed-case identifiers to snake_case."""
        s = name.replace('-', '_').replace(' ', '_')
        out = []
        prev_lower = False
        for ch in s:
            if ch.isupper():
                if prev_lower:
                    out.append('_')
                out.append(ch.lower())
                prev_lower = False
            else:
                out.append(ch)
                prev_lower = ch.isalpha()
        # collapse double underscores
        snake = ''.join(out)
        while '__' in snake:
            snake = snake.replace('__', '_')
        return snake
    
    def _extract_flow_rate(self, flow: Flow) -> str:
        """Extract rate annotation from flow."""
        return (
            getattr(flow, '_flow_rate', None)
            or getattr(flow.__class__, '_flow_rate', None)
            or getattr(flow, '_rate', None)
            or getattr(flow.__class__, '_rate', None)
            or "20hz"
        )
    
    def _convert_rate_to_millis(self, rate: str) -> int:
        """Convert rate string to milliseconds."""
        if not rate:
            return 50  # Default 20Hz
        
        rate = rate.lower()
        if rate.endswith('hz'):
            hz = float(rate[:-2])
            return int(1000 / hz)
        elif rate.endswith('ms'):
            return int(float(rate[:-2]))
        elif rate.endswith('s'):
            return int(float(rate[:-1]) * 1000)
        else:
            try:
                hz = float(rate)
                return int(1000 / hz)
            except ValueError:
                return 50
