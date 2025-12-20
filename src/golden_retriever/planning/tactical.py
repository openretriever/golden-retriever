"""
Tactical Planning Implementation

Implements the tactical planning layer of the bilevel architecture:
- Low-level skill execution planning (10Hz)  
- Real-time action sequence generation (30Hz)
- Skill policy integration (RT-1, π0, custom policies)
- Reactive control with sensor feedback

Tactical planners operate at 10-30Hz and focus on:
- How to execute specific skills
- What actions to take moment-to-moment
- How to react to sensor feedback
- How to adapt to dynamic environments
"""

from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import time

from ..core.types import Flow, Eff, RGBImage
from .bilevel import (
    SkillInstruction, TacticalPlan, RobotAction, RobotState,
    TacticalPlanner as TacticalPlannerInterface,
    SkillPolicy as SkillPolicyInterface
)


# ======================== TACTICAL PLANNING IMPLEMENTATIONS ========================

class PrimitiveActionPlanner(TacticalPlannerInterface):
    """
    Primitive action-based tactical planner.
    
    Decomposes skills into sequences of basic robot actions:
    - Move arm to position
    - Open/close gripper  
    - Navigate to pose
    - Wait/monitor
    """
    
    def __init__(self):
        self.action_library = self._initialize_action_library()
    
    def _initialize_action_library(self) -> Dict[str, Callable]:
        """Initialize library of action generation functions."""
        return {
            "navigate_to_object": self._plan_navigation_to_object,
            "navigate_to_location": self._plan_navigation_to_location,
            "pick_object": self._plan_pick_object,
            "place_object": self._plan_place_object,
            "visual_search": self._plan_visual_search,
            "systematic_search": self._plan_systematic_search,
            "report_status": self._plan_report_status
        }
    
    def run(self, inputs: Tuple[SkillInstruction, RobotState]) -> TacticalPlan:
        """
        Generate tactical plan for specific skill execution.
        
        Converts high-level skill instruction into sequence of primitive actions
        that can be executed by the robot's low-level controllers.
        """
        skill_instruction, robot_state = inputs
        
        # Get action planning function for this skill
        planner_func = self.action_library.get(skill_instruction.skill_name)
        if not planner_func:
            print(f"⚠️ No tactical planner for skill: {skill_instruction.skill_name}")
            return self._create_fallback_plan(skill_instruction)
        
        # Generate action sequence
        actions = planner_func(skill_instruction, robot_state)
        
        # Create tactical plan
        plan = TacticalPlan(
            skill_instruction=skill_instruction,
            action_sequence=actions,
            control_parameters={
                "execution_rate": 30.0,  # 30Hz execution
                "safety_monitoring": True,
                "adaptive_replanning": True
            },
            safety_constraints=[
                "collision_avoidance",
                "joint_limits", 
                "force_limits"
            ]
        )
        
        print(f"⚡ Tactical Plan: {len(actions)} actions for {skill_instruction.skill_name}")
        return plan
    
    def _plan_navigation_to_object(self, skill: SkillInstruction, state: RobotState) -> List[RobotAction]:
        """Plan navigation to object."""
        target_object = skill.parameters.get("object", "unknown")
        approach_distance = skill.parameters.get("approach_distance", 0.5)
        
        # Find object in environment
        target_pose = self._find_object_pose(target_object, state)
        
        return [
            RobotAction(
                action_type="plan_path",
                parameters={
                    "target_pose": target_pose,
                    "approach_distance": approach_distance
                },
                expected_duration=2.0
            ),
            RobotAction(
                action_type="execute_navigation",
                parameters={
                    "path_execution": True,
                    "obstacle_avoidance": True
                },
                expected_duration=8.0,
                safety_constraints=["collision_avoidance", "path_following"]
            )
        ]
    
    def _plan_navigation_to_location(self, skill: SkillInstruction, state: RobotState) -> List[RobotAction]:
        """Plan navigation to specific location."""
        target_location = skill.parameters.get("location")
        orientation = skill.parameters.get("orientation", "auto")
        
        return [
            RobotAction(
                action_type="plan_path",
                parameters={
                    "target_location": target_location,
                    "target_orientation": orientation
                },
                expected_duration=2.0
            ),
            RobotAction(
                action_type="execute_navigation", 
                parameters={
                    "precise_positioning": True
                },
                expected_duration=8.0
            )
        ]
    
    def _plan_pick_object(self, skill: SkillInstruction, state: RobotState) -> List[RobotAction]:
        """Plan object picking sequence."""
        target_object = skill.parameters.get("object")
        approach = skill.parameters.get("approach", "top_down")
        force = skill.parameters.get("force", "gentle")
        
        return [
            RobotAction(
                action_type="visual_locate",
                parameters={
                    "target_object": target_object,
                    "precision": "high"
                },
                expected_duration=2.0
            ),
            RobotAction(
                action_type="plan_grasp",
                parameters={
                    "object": target_object,
                    "approach_vector": approach,
                    "force_profile": force
                },
                expected_duration=1.0
            ),
            RobotAction(
                action_type="approach_object",
                parameters={
                    "approach_speed": "slow",
                    "contact_detection": True
                },
                expected_duration=4.0,
                safety_constraints=["force_limits", "collision_detection"]
            ),
            RobotAction(
                action_type="close_gripper",
                parameters={
                    "force_limit": force,
                    "contact_feedback": True
                },
                expected_duration=2.0,
                safety_constraints=["grip_force_limits"]
            ),
            RobotAction(
                action_type="lift_object",
                parameters={
                    "lift_height": 0.1,
                    "stability_check": True
                },
                expected_duration=3.0,
                safety_constraints=["object_stability"]
            )
        ]
    
    def _plan_place_object(self, skill: SkillInstruction, state: RobotState) -> List[RobotAction]:
        """Plan object placement sequence."""
        target_location = skill.parameters.get("location")
        approach = skill.parameters.get("approach", "gentle")
        precision = skill.parameters.get("precision", "medium")
        
        return [
            RobotAction(
                action_type="navigate_above_location",
                parameters={
                    "target_location": target_location,
                    "clearance_height": 0.2
                },
                expected_duration=3.0
            ),
            RobotAction(
                action_type="lower_object",
                parameters={
                    "descent_speed": "slow",
                    "contact_detection": True,
                    "precision_level": precision
                },
                expected_duration=4.0,
                safety_constraints=["contact_detection", "gentle_placement"]
            ),
            RobotAction(
                action_type="open_gripper",
                parameters={
                    "release_speed": "slow"
                },
                expected_duration=1.0
            ),
            RobotAction(
                action_type="retract_arm",
                parameters={
                    "retract_distance": 0.1,
                    "safety_clearance": True
                },
                expected_duration=2.0
            )
        ]
    
    def _plan_visual_search(self, skill: SkillInstruction, state: RobotState) -> List[RobotAction]:
        """Plan visual search sequence."""
        search_query = skill.parameters.get("search_query", "objects")
        search_area = skill.parameters.get("search_area", "workspace")
        
        return [
            RobotAction(
                action_type="position_camera",
                parameters={
                    "search_area": search_area,
                    "optimal_viewpoint": True
                },
                expected_duration=3.0
            ),
            RobotAction(
                action_type="scan_environment",
                parameters={
                    "search_pattern": "systematic",
                    "detection_query": search_query
                },
                expected_duration=5.0
            )
        ]
    
    def _plan_systematic_search(self, skill: SkillInstruction, state: RobotState) -> List[RobotAction]:
        """Plan systematic exploration sequence."""
        area = skill.parameters.get("area", "workspace")
        search_pattern = skill.parameters.get("search_pattern", "grid")
        
        return [
            RobotAction(
                action_type="generate_search_pattern",
                parameters={
                    "area": area,
                    "pattern_type": search_pattern
                },
                expected_duration=1.0
            ),
            RobotAction(
                action_type="execute_search_pattern",
                parameters={
                    "systematic_coverage": True,
                    "adaptive_timing": True
                },
                expected_duration=12.0
            )
        ]
    
    def _plan_report_status(self, skill: SkillInstruction, state: RobotState) -> List[RobotAction]:
        """Plan status reporting."""
        message = skill.parameters.get("message", "status_update")
        
        return [
            RobotAction(
                action_type="generate_report",
                parameters={
                    "message": message,
                    "include_state": True
                },
                expected_duration=1.0
            )
        ]
    
    def _find_object_pose(self, object_name: str, state: RobotState) -> Dict[str, float]:
        """Find pose of object in environment (simplified)."""
        # In real implementation, this would use perception systems
        for obj in state.environment_objects:
            if obj.get("name") == object_name:
                return obj.get("pose", {"x": 1.0, "y": 0.0, "z": 0.0})
        
        # Default pose if object not found
        return {"x": 1.0, "y": 0.0, "z": 0.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    
    def _create_fallback_plan(self, skill: SkillInstruction) -> TacticalPlan:
        """Create fallback plan for unknown skills."""
        fallback_action = RobotAction(
            action_type="report_error",
            parameters={
                "error": f"Unknown skill: {skill.skill_name}",
                "skill_parameters": skill.parameters
            },
            expected_duration=1.0
        )
        
        return TacticalPlan(
            skill_instruction=skill,
            action_sequence=[fallback_action],
            control_parameters={"error_handling": True},
            safety_constraints=["error_recovery"]
        )


# ======================== SKILL POLICY IMPLEMENTATIONS ========================

class RT1SkillPolicy(SkillPolicyInterface):
    """
    RT-1 (Robotics Transformer) based skill policy.
    
    Implements real-time action generation using RT-1 vision-language-action model
    for reactive control based on visual observations and language instructions.
    """
    
    def __init__(self, model_path: str = "rt1_x_jax"):
        self.model_path = model_path
        self._initialize_rt1()
    
    def _initialize_rt1(self):
        """Initialize RT-1 model."""
        print(f"🤖 Initializing RT-1 skill policy: {self.model_path}")
        # In real implementation, this would load the actual RT-1 model
        self.rt1_model = None
        self.action_tokenizer = None
    
    def run(self, inputs: Tuple[RobotState, TacticalPlan]) -> Eff[RobotState, RobotAction]:
        """
        Generate next action using RT-1 policy.
        
        Uses RT-1's vision-language-action capabilities to generate contextually 
        appropriate actions based on current visual observations and tactical plan.
        """
        robot_state, tactical_plan = inputs
        
        def rt1_action_generation(state: RobotState) -> Tuple[RobotAction, RobotState]:
            # Extract visual observation
            visual_obs = self._extract_visual_observation(state)
            
            # Extract language instruction from tactical plan
            language_instruction = self._extract_language_instruction(tactical_plan)
            
            # Generate action using RT-1
            action = self._rt1_predict_action(visual_obs, language_instruction, state)
            
            # Update state to track RT-1 execution
            new_state = state._replace(
                current_skill=f"rt1_{tactical_plan.skill_instruction.skill_name}",
                last_action_result={"policy": "rt1", "timestamp": time.time()}
            )
            
            return action, new_state
        
        return Eff(rt1_action_generation)
    
    def _extract_visual_observation(self, state: RobotState) -> np.ndarray:
        """Extract visual observation for RT-1 input."""
        # In real implementation, this would get current camera image
        # For now, return placeholder observation
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _extract_language_instruction(self, tactical_plan: TacticalPlan) -> str:
        """Extract language instruction for RT-1."""
        skill = tactical_plan.skill_instruction
        return f"{skill.skill_name} with {skill.parameters}"
    
    def _rt1_predict_action(self, visual_obs: np.ndarray, instruction: str, state: RobotState) -> RobotAction:
        """Use RT-1 model to predict next action."""
        # In real implementation, this would:
        # 1. Preprocess visual observation
        # 2. Tokenize language instruction  
        # 3. Run RT-1 inference
        # 4. Post-process action tokens to robot actions
        
        # For now, return action based on current tactical plan
        return RobotAction(
            action_type="rt1_action",
            parameters={
                "instruction": instruction,
                "visual_context": "processed",
                "model_output": "simulated"
            },
            expected_duration=0.033  # 30Hz execution
        )


class Pi0SkillPolicy(SkillPolicyInterface):
    """
    π0 (Pi-Zero) foundation model based skill policy.
    
    Uses Physical Intelligence's π0 foundation model for general robotic control
    with broad capability across manipulation, navigation, and interaction tasks.
    """
    
    def __init__(self, model_path: str = "pi0_foundation"):
        self.model_path = model_path
        self._initialize_pi0()
    
    def _initialize_pi0(self):
        """Initialize π0 foundation model."""
        print(f"🧠 Initializing π0 skill policy: {self.model_path}")
        self.pi0_model = None
    
    def run(self, inputs: Tuple[RobotState, TacticalPlan]) -> Eff[RobotState, RobotAction]:
        """Generate next action using π0 foundation model."""
        robot_state, tactical_plan = inputs
        
        def pi0_action_generation(state: RobotState) -> Tuple[RobotAction, RobotState]:
            # Create multimodal observation for π0
            observation = self._create_pi0_observation(state, tactical_plan)
            
            # Generate action using π0
            action = self._pi0_predict_action(observation, state)
            
            # Update state
            new_state = state._replace(
                current_skill=f"pi0_{tactical_plan.skill_instruction.skill_name}",
                last_action_result={"policy": "pi0", "timestamp": time.time()}
            )
            
            return action, new_state
        
        return Eff(pi0_action_generation)
    
    def _create_pi0_observation(self, state: RobotState, tactical_plan: TacticalPlan) -> Dict[str, Any]:
        """Create multimodal observation for π0."""
        return {
            "visual": np.zeros((224, 224, 3)),  # Camera image
            "proprioception": state.joint_positions or [0.0] * 7,  # Joint states
            "task_description": tactical_plan.skill_instruction.skill_name,
            "environment_context": state.environment_objects
        }
    
    def _pi0_predict_action(self, observation: Dict[str, Any], state: RobotState) -> RobotAction:
        """Use π0 to predict next action."""
        # π0 inference would happen here
        return RobotAction(
            action_type="pi0_action",
            parameters={
                "task": observation["task_description"],
                "multimodal_input": "processed",
                "foundation_reasoning": True
            },
            expected_duration=0.033
        )


class CustomSkillPolicy(SkillPolicyInterface):
    """
    Custom skill policy for domain-specific or hand-crafted behaviors.
    
    Implements task-specific control logic for skills that require:
    - Precise control (e.g., fine manipulation)
    - Safety-critical operations
    - Domain-specific expertise
    - Legacy system integration
    """
    
    def __init__(self, skill_implementations: Dict[str, Callable] = None):
        self.skill_implementations = skill_implementations or {}
        self._register_default_skills()
    
    def _register_default_skills(self):
        """Register default skill implementations."""
        self.skill_implementations.update({
            "precise_placement": self._precise_placement_policy,
            "force_controlled_insertion": self._force_controlled_policy,
            "safety_monitoring": self._safety_monitoring_policy,
            "calibration_routine": self._calibration_policy
        })
    
    def run(self, inputs: Tuple[RobotState, TacticalPlan]) -> Eff[RobotState, RobotAction]:
        """Execute custom skill policy."""
        robot_state, tactical_plan = inputs
        
        skill_name = tactical_plan.skill_instruction.skill_name
        skill_impl = self.skill_implementations.get(skill_name)
        
        if not skill_impl:
            # Fall back to generic action
            return self._generic_skill_policy(inputs)
        
        def custom_skill_execution(state: RobotState) -> Tuple[RobotAction, RobotState]:
            action = skill_impl(state, tactical_plan)
            new_state = state._replace(
                current_skill=f"custom_{skill_name}",
                last_action_result={"policy": "custom", "timestamp": time.time()}
            )
            return action, new_state
        
        return Eff(custom_skill_execution)
    
    def _precise_placement_policy(self, state: RobotState, plan: TacticalPlan) -> RobotAction:
        """Custom policy for precise object placement."""
        return RobotAction(
            action_type="precise_position_control",
            parameters={
                "position_tolerance": 0.001,  # 1mm precision
                "force_feedback": True,
                "compliance_control": True
            },
            expected_duration=2.0,
            safety_constraints=["force_limits", "position_limits"]
        )
    
    def _force_controlled_policy(self, state: RobotState, plan: TacticalPlan) -> RobotAction:
        """Custom policy for force-controlled operations."""
        return RobotAction(
            action_type="force_controlled_motion",
            parameters={
                "force_limit": 5.0,  # 5N maximum force
                "compliance_matrix": "diagonal",
                "contact_detection": True
            },
            expected_duration=3.0,
            safety_constraints=["force_monitoring", "emergency_stop"]
        )
    
    def _safety_monitoring_policy(self, state: RobotState, plan: TacticalPlan) -> RobotAction:
        """Custom policy for safety monitoring."""
        return RobotAction(
            action_type="safety_check",
            parameters={
                "collision_detection": True,
                "workspace_limits": True,
                "emergency_procedures": "enabled"
            },
            expected_duration=0.1,
            safety_constraints=["continuous_monitoring"]
        )
    
    def _calibration_policy(self, state: RobotState, plan: TacticalPlan) -> RobotAction:
        """Custom policy for calibration routines."""
        return RobotAction(
            action_type="calibration_sequence",
            parameters={
                "calibration_type": "tactile_sensors",
                "reference_points": "predefined",
                "accuracy_target": 0.01
            },
            expected_duration=10.0
        )
    
    def _generic_skill_policy(self, inputs: Tuple[RobotState, TacticalPlan]) -> Eff[RobotState, RobotAction]:
        """Generic fallback skill policy."""
        robot_state, tactical_plan = inputs
        
        def generic_execution(state: RobotState) -> Tuple[RobotAction, RobotState]:
            action = RobotAction(
                action_type="generic_execution",
                parameters=tactical_plan.skill_instruction.parameters,
                expected_duration=1.0
            )
            new_state = state._replace(
                current_skill=f"generic_{tactical_plan.skill_instruction.skill_name}"
            )
            return action, new_state
        
        return Eff(generic_execution)


# ======================== FACTORY FUNCTIONS ========================

def create_primitive_tactical_planner() -> PrimitiveActionPlanner:
    """Create primitive action-based tactical planner."""
    return PrimitiveActionPlanner()


def create_rt1_skill_policy(model_path: str = "rt1_x_jax") -> RT1SkillPolicy:
    """Create RT-1 based skill policy."""
    return RT1SkillPolicy(model_path)


def create_pi0_skill_policy(model_path: str = "pi0_foundation") -> Pi0SkillPolicy:
    """Create π0 foundation model based skill policy."""
    return Pi0SkillPolicy(model_path)


def create_custom_skill_policy(skill_implementations: Dict[str, Callable] = None) -> CustomSkillPolicy:
    """Create custom skill policy with domain-specific implementations."""
    return CustomSkillPolicy(skill_implementations)


def create_hybrid_skill_policy(
    primary_policy: SkillPolicyInterface,
    fallback_policy: SkillPolicyInterface
) -> SkillPolicyInterface:
    """Create hybrid skill policy that falls back if primary fails."""
    
    class HybridSkillPolicy(SkillPolicyInterface):
        def __init__(self, primary: SkillPolicyInterface, fallback: SkillPolicyInterface):
            self.primary = primary
            self.fallback = fallback
        
        def run(self, inputs: Tuple[RobotState, TacticalPlan]) -> Eff[RobotState, RobotAction]:
            def hybrid_execution(state: RobotState) -> Tuple[RobotAction, RobotState]:
                try:
                    # Try primary policy first
                    return self.primary.run(inputs).run(state)
                except Exception as e:
                    print(f"⚠️ Primary policy failed: {e}, using fallback")
                    return self.fallback.run(inputs).run(state)
            
            return Eff(hybrid_execution)
    
    return HybridSkillPolicy(primary_policy, fallback_policy)