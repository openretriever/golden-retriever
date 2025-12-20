"""
Strategic Planning Implementation

Implements the strategic planning layer of the bilevel architecture:
- High-level task decomposition (1Hz)
- Skill selection and sequencing  
- Goal-oriented planning with constraints
- Integration with VLMs and LLMs for natural language understanding

Strategic planners operate at 1Hz and focus on:
- What skills need to be executed
- In what order
- With what high-level parameters
- Under what constraints
"""

from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import json
import time

from ..core.types import Flow
from .bilevel import (
    TaskRequest, StrategicPlan, SkillInstruction, RobotState, 
    StrategicPlanner as StrategicPlannerInterface
)


@dataclass
class SkillLibrary:
    """Registry of available skills for strategic planning."""
    
    skills: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def register_skill(self, name: str, description: str, parameters: List[str], 
                      preconditions: List[str] = None, postconditions: List[str] = None):
        """Register a skill in the library."""
        self.skills[name] = {
            "name": name,
            "description": description, 
            "parameters": parameters,
            "preconditions": preconditions or [],
            "postconditions": postconditions or []
        }
    
    def get_skill_descriptions(self) -> str:
        """Get formatted skill descriptions for LLM prompting."""
        descriptions = []
        for skill_name, skill_info in self.skills.items():
            desc = f"- {skill_name}: {skill_info['description']}"
            if skill_info['parameters']:
                desc += f" (params: {', '.join(skill_info['parameters'])})"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def validate_skill(self, skill_name: str) -> bool:
        """Check if skill exists in library."""
        return skill_name in self.skills


class LLMStrategicPlanner(StrategicPlannerInterface):
    """
    LLM-based strategic planner for natural language task decomposition.
    
    Uses large language models to:
    1. Parse natural language task requests
    2. Decompose tasks into skill sequences
    3. Generate appropriate parameters for each skill
    4. Consider preconditions and constraints
    """
    
    def __init__(self, skill_library: SkillLibrary, model_name: str = "gpt-4"):
        self.skill_library = skill_library
        self.model_name = model_name
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM connection (placeholder for actual implementation)."""
        # In real implementation, this would initialize the LLM client
        print(f"🧠 Initializing strategic planner with {self.model_name}")
        self.llm_client = None  # Placeholder
    
    def run(self, inputs: Tuple[TaskRequest, RobotState]) -> StrategicPlan:
        """
        Create strategic plan from task request and current robot state.
        
        This is the core strategic planning logic that:
        1. Analyzes the task request and current state
        2. Decomposes the task into a sequence of skills
        3. Generates parameters for each skill
        4. Creates a complete strategic plan
        """
        task_request, robot_state = inputs
        
        # Generate planning prompt
        prompt = self._create_planning_prompt(task_request, robot_state)
        
        # Get LLM response (simulated for now)
        llm_response = self._query_llm(prompt)
        
        # Parse response into skill sequence
        skills = self._parse_skill_sequence(llm_response, task_request)
        
        # Create strategic plan
        plan = StrategicPlan(
            task_id=f"task_{int(time.time())}",
            skills=skills,
            goal_state=task_request.goal_state or {},
            estimated_duration=sum(skill.timeout for skill in skills),
            constraints=task_request.constraints
        )
        
        print(f"🎯 Strategic Plan Created: {len(skills)} skills, {plan.estimated_duration:.1f}s estimated")
        for i, skill in enumerate(skills):
            print(f"  {i+1}. {skill.skill_name}: {skill.parameters}")
        
        return plan
    
    def _create_planning_prompt(self, task_request: TaskRequest, robot_state: RobotState) -> str:
        """Create LLM prompt for strategic planning."""
        
        # Current environment description
        environment_desc = self._describe_environment(robot_state)
        
        # Available skills
        skills_desc = self.skill_library.get_skill_descriptions()
        
        prompt = f"""You are a strategic planner for a mobile manipulation robot. 
Given a task request and current state, create a sequence of skills to accomplish the task.

TASK REQUEST:
{task_request.description}

CURRENT ROBOT STATE:
{environment_desc}

AVAILABLE SKILLS:
{skills_desc}

CONSTRAINTS:
{'; '.join(task_request.constraints) if task_request.constraints else 'None'}

Please provide a skill sequence as a JSON list where each skill has:
- skill_name: name from available skills
- parameters: dict of parameters for the skill
- timeout: estimated time in seconds

Example format:
[
  {{"skill_name": "navigate_to", "parameters": {{"target": "kitchen_table"}}, "timeout": 10.0}},
  {{"skill_name": "pick_object", "parameters": {{"object": "red_cup"}}, "timeout": 15.0}}
]

Skill sequence:"""
        
        return prompt
    
    def _describe_environment(self, robot_state: RobotState) -> str:
        """Create natural language description of current environment."""
        descriptions = []
        
        if robot_state.pose:
            descriptions.append(f"Robot position: {robot_state.pose}")
        
        if robot_state.held_objects:
            descriptions.append(f"Holding: {', '.join(robot_state.held_objects)}")
        else:
            descriptions.append("Hands empty")
        
        if robot_state.environment_objects:
            obj_names = [obj.get('name', 'unknown') for obj in robot_state.environment_objects]
            descriptions.append(f"Visible objects: {', '.join(obj_names)}")
        
        if robot_state.current_skill:
            descriptions.append(f"Currently executing: {robot_state.current_skill}")
        
        return "; ".join(descriptions)
    
    def _query_llm(self, prompt: str) -> str:
        """Query LLM with planning prompt (simulated)."""
        # In real implementation, this would call the actual LLM
        # For now, return a simulated response based on common patterns
        
        if "pick up" in prompt.lower() or "grasp" in prompt.lower():
            return '''[
  {"skill_name": "navigate_to_object", "parameters": {"object": "target_object"}, "timeout": 10.0},
  {"skill_name": "pick_object", "parameters": {"object": "target_object", "approach": "top_down"}, "timeout": 15.0}
]'''
        
        elif "place" in prompt.lower() or "put" in prompt.lower():
            return '''[
  {"skill_name": "navigate_to_location", "parameters": {"location": "target_location"}, "timeout": 10.0},
  {"skill_name": "place_object", "parameters": {"location": "target_location", "approach": "gentle"}, "timeout": 12.0}
]'''
        
        else:
            # Generic exploration/navigation task
            return '''[
  {"skill_name": "explore_area", "parameters": {"area": "workspace"}, "timeout": 20.0},
  {"skill_name": "report_status", "parameters": {"message": "task_completed"}, "timeout": 2.0}
]'''
    
    def _parse_skill_sequence(self, llm_response: str, task_request: TaskRequest) -> List[SkillInstruction]:
        """Parse LLM response into SkillInstruction objects."""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON array found in LLM response")
            
            skill_data = json.loads(json_match.group())
            
            skills = []
            for item in skill_data:
                # Validate skill exists
                skill_name = item.get("skill_name")
                if not self.skill_library.validate_skill(skill_name):
                    print(f"⚠️ Unknown skill '{skill_name}', skipping")
                    continue
                
                # Create skill instruction
                skill = SkillInstruction(
                    skill_name=skill_name,
                    parameters=item.get("parameters", {}),
                    timeout=item.get("timeout", 30.0),
                    preconditions=[],  # Could be extracted from skill library
                    postconditions=[]  # Could be extracted from skill library
                )
                skills.append(skill)
            
            return skills
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"⚠️ Failed to parse LLM response: {e}")
            print(f"Response was: {llm_response}")
            
            # Return fallback plan
            return [
                SkillInstruction(
                    skill_name="report_status",
                    parameters={"message": "planning_failed", "error": str(e)},
                    timeout=5.0
                )
            ]


class VLAStrategicPlanner(StrategicPlannerInterface):
    """
    Vision-Language-Action model based strategic planner.
    
    Uses VLA models (like RT-1, π0) for strategic planning that considers:
    1. Visual understanding of the current scene
    2. Language understanding of the task
    3. Action-level reasoning for skill selection
    """
    
    def __init__(self, skill_library: SkillLibrary, model_name: str = "rt1"):
        self.skill_library = skill_library
        self.model_name = model_name
        self._initialize_vla()
    
    def _initialize_vla(self):
        """Initialize VLA model connection."""
        print(f"👁️ Initializing VLA strategic planner with {self.model_name}")
        self.vla_model = None  # Placeholder
    
    def run(self, inputs: Tuple[TaskRequest, RobotState]) -> StrategicPlan:
        """
        Create strategic plan using VLA model reasoning.
        
        This integrates visual scene understanding with language task description
        to generate appropriate skill sequences.
        """
        task_request, robot_state = inputs
        
        # In real implementation, this would:
        # 1. Extract visual features from robot's camera
        # 2. Encode task description as language features
        # 3. Use VLA model to predict skill sequence
        # 4. Convert model outputs to SkillInstruction objects
        
        # For now, provide a simplified implementation
        skills = self._generate_vla_based_skills(task_request, robot_state)
        
        plan = StrategicPlan(
            task_id=f"vla_task_{int(time.time())}",
            skills=skills,
            goal_state=task_request.goal_state or {},
            estimated_duration=sum(skill.timeout for skill in skills),
            constraints=task_request.constraints
        )
        
        print(f"👁️ VLA Strategic Plan: {len(skills)} skills from vision-language reasoning")
        return plan
    
    def _generate_vla_based_skills(self, task_request: TaskRequest, robot_state: RobotState) -> List[SkillInstruction]:
        """Generate skills using VLA model (simplified implementation)."""
        
        # This would use actual VLA model inference
        # For now, create skills based on simple heuristics
        
        if "pick" in task_request.description.lower():
            return [
                SkillInstruction(
                    skill_name="visual_search",
                    parameters={"search_query": "graspable_object"},
                    timeout=8.0
                ),
                SkillInstruction(
                    skill_name="approach_object", 
                    parameters={"approach_strategy": "vision_guided"},
                    timeout=12.0
                ),
                SkillInstruction(
                    skill_name="grasp_object",
                    parameters={"grasp_type": "precision", "force": "gentle"},
                    timeout=10.0
                )
            ]
        
        else:
            return [
                SkillInstruction(
                    skill_name="visual_exploration",
                    parameters={"exploration_type": "systematic"},
                    timeout=15.0
                )
            ]


class TemplateStrategicPlanner(StrategicPlannerInterface):
    """
    Template-based strategic planner for predefined task patterns.
    
    Uses predefined templates for common robotics tasks:
    - Pick and place operations
    - Navigation tasks  
    - Exploration missions
    - Cleaning operations
    """
    
    def __init__(self, skill_library: SkillLibrary):
        self.skill_library = skill_library
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize predefined task templates."""
        return {
            "pick_and_place": [
                {"skill_name": "navigate_to_object", "timeout": 10.0},
                {"skill_name": "pick_object", "timeout": 15.0},
                {"skill_name": "navigate_to_location", "timeout": 10.0},
                {"skill_name": "place_object", "timeout": 12.0}
            ],
            "exploration": [
                {"skill_name": "systematic_search", "timeout": 30.0},
                {"skill_name": "map_environment", "timeout": 20.0},
                {"skill_name": "report_findings", "timeout": 5.0}
            ],
            "cleaning": [
                {"skill_name": "detect_debris", "timeout": 10.0},
                {"skill_name": "navigate_to_debris", "timeout": 8.0},
                {"skill_name": "clean_area", "timeout": 20.0},
                {"skill_name": "dispose_debris", "timeout": 10.0}
            ]
        }
    
    def run(self, inputs: Tuple[TaskRequest, RobotState]) -> StrategicPlan:
        """Create strategic plan from predefined templates."""
        task_request, robot_state = inputs
        
        # Match task to template
        template_name = self._match_task_to_template(task_request.description)
        template = self.templates.get(template_name, self.templates["exploration"])
        
        # Instantiate template with task-specific parameters
        skills = self._instantiate_template(template, task_request, robot_state)
        
        plan = StrategicPlan(
            task_id=f"template_task_{int(time.time())}",
            skills=skills,
            goal_state=task_request.goal_state or {},
            estimated_duration=sum(skill.timeout for skill in skills),
            constraints=task_request.constraints
        )
        
        print(f"📋 Template Strategic Plan ({template_name}): {len(skills)} skills")
        return plan
    
    def _match_task_to_template(self, description: str) -> str:
        """Match task description to appropriate template."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["pick", "grasp", "place", "put"]):
            return "pick_and_place"
        elif any(word in description_lower for word in ["clean", "tidy", "organize"]):
            return "cleaning"
        elif any(word in description_lower for word in ["explore", "search", "find", "look"]):
            return "exploration"
        else:
            return "exploration"  # Default template
    
    def _instantiate_template(self, template: List[Dict[str, Any]], 
                            task_request: TaskRequest, robot_state: RobotState) -> List[SkillInstruction]:
        """Instantiate template with specific parameters."""
        skills = []
        
        for skill_template in template:
            # Extract basic parameters from task request and state
            parameters = self._extract_parameters(skill_template, task_request, robot_state)
            
            skill = SkillInstruction(
                skill_name=skill_template["skill_name"],
                parameters=parameters,
                timeout=skill_template.get("timeout", 30.0)
            )
            skills.append(skill)
        
        return skills
    
    def _extract_parameters(self, skill_template: Dict[str, Any], 
                          task_request: TaskRequest, robot_state: RobotState) -> Dict[str, Any]:
        """Extract skill parameters from task and state."""
        # This would implement sophisticated parameter extraction
        # For now, return basic parameters
        
        base_params = skill_template.get("parameters", {})
        
        # Add task-specific information
        if "object" in task_request.description:
            # Try to extract object name from description
            words = task_request.description.split()
            for i, word in enumerate(words):
                if word.lower() in ["pick", "grasp", "get"]:
                    if i + 1 < len(words):
                        base_params["target_object"] = words[i + 1]
                        break
        
        return base_params


# ======================== FACTORY FUNCTIONS ========================

def create_default_skill_library() -> SkillLibrary:
    """Create a default skill library with common robotics skills."""
    library = SkillLibrary()
    
    # Navigation skills
    library.register_skill(
        "navigate_to_object",
        "Navigate to a specific object in the environment",
        ["object", "approach_distance"],
        preconditions=["object_visible"],
        postconditions=["near_object"]
    )
    
    library.register_skill(
        "navigate_to_location", 
        "Navigate to a specific location/pose",
        ["location", "orientation"],
        postconditions=["at_location"]
    )
    
    # Manipulation skills
    library.register_skill(
        "pick_object",
        "Pick up an object using appropriate grasp",
        ["object", "approach", "force"],
        preconditions=["near_object", "gripper_empty"],
        postconditions=["holding_object"]
    )
    
    library.register_skill(
        "place_object",
        "Place held object at specified location",
        ["location", "approach", "precision"],
        preconditions=["holding_object", "at_location"],
        postconditions=["object_placed", "gripper_empty"]
    )
    
    # Perception skills
    library.register_skill(
        "visual_search",
        "Search for objects matching visual criteria",
        ["search_query", "search_area"],
        postconditions=["objects_detected"]
    )
    
    library.register_skill(
        "systematic_search",
        "Systematically explore and map environment", 
        ["area", "search_pattern"],
        postconditions=["area_mapped"]
    )
    
    # Utility skills
    library.register_skill(
        "report_status",
        "Report current status or completion message",
        ["message", "recipient"],
        postconditions=["status_reported"]
    )
    
    return library


def create_llm_strategic_planner(model_name: str = "gpt-4") -> LLMStrategicPlanner:
    """Create LLM-based strategic planner with default skill library."""
    skill_library = create_default_skill_library()
    return LLMStrategicPlanner(skill_library, model_name)


def create_vla_strategic_planner(model_name: str = "rt1") -> VLAStrategicPlanner:
    """Create VLA-based strategic planner with default skill library.""" 
    skill_library = create_default_skill_library()
    return VLAStrategicPlanner(skill_library, model_name)


def create_template_strategic_planner() -> TemplateStrategicPlanner:
    """Create template-based strategic planner with default skill library."""
    skill_library = create_default_skill_library()
    return TemplateStrategicPlanner(skill_library)