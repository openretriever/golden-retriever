"""
Retriever Planning System - Clean Bilevel Architecture

This package provides the complete bilevel closed-loop planning architecture 
from Proposal v2.7, built on validated single-level patterns.

Key Features:
- Strategic Planning (1Hz): Task decomposition and skill selection
- Tactical Planning (10Hz): Action sequence generation  
- Skill Execution (30Hz): Real-time reactive control
- FRP Coordination: Automatic temporal coordination
- Closed-Loop Control: Continuous monitoring and replanning

Architecture:
- Single-Level: Simple closed-loop for basic planning + execution + monitoring
- Bilevel: Full strategic/tactical separation with automatic coordination
- Production: Dora FRP backend integration for real-time deployment
"""

# Bilevel closed-loop planning architecture
from .bilevel import (
    # Core types
    TaskRequest,
    StrategicPlan,
    SkillInstruction, 
    TacticalPlan,
    RobotAction,
    RobotState as BilevelRobotState,
    ExecutionResult,
    PlanningResult,
    
    # Interfaces
    StrategicPlanner,
    TacticalPlanner,
    SkillPolicy,
    PlanningMonitor,
    
    # Main system
    BilevelPlanningSystem,
    ClosedLoopPlanningFlow,
    create_bilevel_system,
    create_default_monitor
)

from .strategic import (
    # Strategic planners
    LLMStrategicPlanner,
    VLAStrategicPlanner,
    TemplateStrategicPlanner,
    SkillLibrary,
    
    # Factory functions
    create_llm_strategic_planner,
    create_vla_strategic_planner, 
    create_template_strategic_planner,
    create_default_skill_library
)

from .tactical import (
    # Tactical planners and policies
    PrimitiveActionPlanner,
    RT1SkillPolicy,
    Pi0SkillPolicy, 
    CustomSkillPolicy,
    
    # Factory functions
    create_primitive_tactical_planner,
    create_rt1_skill_policy,
    create_pi0_skill_policy,
    create_custom_skill_policy,
    create_hybrid_skill_policy
)

__all__ = [
    # Bilevel planning core types
    "TaskRequest",
    "StrategicPlan", 
    "SkillInstruction",
    "TacticalPlan",
    "RobotAction",
    "BilevelRobotState",
    "ExecutionResult",
    "PlanningResult",
    
    # Bilevel planning interfaces
    "StrategicPlanner",
    "TacticalPlanner", 
    "SkillPolicy",
    "PlanningMonitor",
    
    # Bilevel planning system
    "BilevelPlanningSystem",
    "ClosedLoopPlanningFlow",
    "create_bilevel_system",
    "create_default_monitor",
    
    # Strategic planning
    "LLMStrategicPlanner",
    "VLAStrategicPlanner",
    "TemplateStrategicPlanner", 
    "SkillLibrary",
    "create_llm_strategic_planner",
    "create_vla_strategic_planner",
    "create_template_strategic_planner",
    "create_default_skill_library",
    
    # Tactical planning
    "PrimitiveActionPlanner",
    "RT1SkillPolicy",
    "Pi0SkillPolicy",
    "CustomSkillPolicy", 
    "create_primitive_tactical_planner",
    "create_rt1_skill_policy",
    "create_pi0_skill_policy", 
    "create_custom_skill_policy",
    "create_hybrid_skill_policy"
]