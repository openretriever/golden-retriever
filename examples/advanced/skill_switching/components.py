import time
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from retriever.flow import Flow, io


# =============================================================================
# TYPES
# =============================================================================

class SkillMode(str, Enum):
    IDLE = "idle"
    APPROACH = "approach"
    MANIPULATE = "manipulate"


@io
@dataclass
class RobotState:
    """Simulated state of the robot."""
    x: float
    y: float
    gripper_open: bool = True
    holding_object: bool = False


@io
@dataclass
class UserCommand:
    """Command from the 'user' (or high-level planner) to change modes."""
    mode: str  # One of SkillMode values
    target_x: float = 0.0
    target_y: float = 0.0


@io
@dataclass
class SkillInput:
    """Standardized input packet for a skill."""
    state: RobotState
    target_x: float
    target_y: float
    active: bool = True


@io
@dataclass
class SkillSignal:
    """Wrapper to pass SkillInput as a single object (port) rather than flattened fields."""
    flow: Optional[SkillInput]


@io
@dataclass
class RobotAction:
    """Action emitted by a skill."""
    # Velocity control
    vx: float = 0.0
    vy: float = 0.0
    # Gripper control
    set_gripper_open: Optional[bool] = None
    
    # Debug/Info
    skill_name: str = "unknown"


@io
@dataclass
class ActionSignal:
    """Wrapper for RobotAction."""
    packet: Optional[RobotAction] = None


@io
@dataclass
class RouterInputHelper:
    """
    Flattened input containing all potential fields from upstream.
    We just list them.
    """
    # From RobotState
    x: float = 0.0
    y: float = 0.0
    gripper_open: bool = True
    holding_object: bool = False
    
    # From UserCommand
    mode: str = ""
    target_x: float = 0.0
    target_y: float = 0.0


@io
@dataclass
class RouterOutput:
    """
    Outputs routed signals.
    """
    approach_flow: Optional[SkillInput] = None
    manipulate_flow: Optional[SkillInput] = None
    
    # Debug
    current_mode: str = SkillMode.IDLE.value


@io
@dataclass
class ArbiterInput:
    """Inputs from start wrappers."""
    approach_packet: Optional[RobotAction] = None
    manipulate_packet: Optional[RobotAction] = None


# =============================================================================
# FLOWS
# =============================================================================

class RobotEnv(Flow[ActionSignal, RobotState]):
    """
    Mock robot environment.
    Integrates velocity to update position.
    """
    HZ = 10.0
    DT = 1.0 / HZ

    def init(self):
        self.state = RobotState(x=0.0, y=0.0)
        self.tick = 0

    def run(self, input: Optional[ActionSignal]) -> RobotState:
        action = input.packet if input else None
        
        # Simple Euler integration
        if action:
            self.state.x += action.vx * self.DT
            self.state.y += action.vy * self.DT
            
            if action.set_gripper_open is not None:
                self.state.gripper_open = action.set_gripper_open
                if not self.state.gripper_open:
                     # Simple logic: success if closed
                     self.state.holding_object = True
                else:
                    self.state.holding_object = False
            
            # Log action
            print(f"[Env] t={self.tick*self.DT:.2f} State={self.state} Action from {action.skill_name}: "
                  f"vx={action.vx:.2f} vy={action.vy:.2f} grip={action.set_gripper_open}")
        else:
             print(f"[Env] t={self.tick*self.DT:.2f} State={self.state} (No Action)")

        self.tick += 1
        return self.state


class Commander(Flow[None, UserCommand]):
    """
    Scripted commander.
    """
    def init(self):
        self.start_time = time.time()

    def run(self, input: None) -> UserCommand:
        elapsed = time.time() - self.start_time
        # print(f"[Commander] t={elapsed:.2f}")
        
        mode = SkillMode.IDLE
        target_x, target_y = 0.0, 0.0
        
        if elapsed < 5.0:
            mode = SkillMode.APPROACH
            target_x, target_y = 5.0, 5.0
        elif elapsed < 8.0:
            mode = SkillMode.MANIPULATE
            target_x, target_y = 5.0, 5.0
        elif elapsed < 12.0:
            mode = SkillMode.APPROACH
            target_x, target_y = 0.0, 0.0
        else:
            mode = SkillMode.IDLE

        return UserCommand(mode=mode.value, target_x=target_x, target_y=target_y)


class SkillRouter(Flow[RouterInputHelper, RouterOutput]):
    
    def init(self):
        self.mode = SkillMode.IDLE
        # Cache command state
        self.cmd_target_x = 0.0
        self.cmd_target_y = 0.0

    def run(self, input: RouterInputHelper) -> RouterOutput:
        print(f"[Router] Input: {input}")
        # Reconstruct State
        state = RobotState(
            x=input.x,
            y=input.y,
            gripper_open=input.gripper_open,
            holding_object=input.holding_object
        )
        
        if input.mode:
            try:
                self.mode = SkillMode(input.mode)
            except ValueError:
                pass
            self.cmd_target_x = input.target_x
            self.cmd_target_y = input.target_y
        
        # Construct Payloads
        active_packet = SkillInput(
            state=state,
            target_x=self.cmd_target_x,
            target_y=self.cmd_target_y,
            active=True
        )
        idle_packet = SkillInput(
            state=state,
            target_x=self.cmd_target_x,
            target_y=self.cmd_target_y,
            active=False
        )
        
        output = RouterOutput(current_mode=self.mode.value)
        
        # Explicitly drive both outputs to ensure downstream gets update
        if self.mode == SkillMode.APPROACH:
            output.approach_flow = active_packet
            output.manipulate_flow = idle_packet
        elif self.mode == SkillMode.MANIPULATE:
            output.approach_flow = idle_packet
            output.manipulate_flow = active_packet
        else:
            # Idle
            output.approach_flow = idle_packet
            output.manipulate_flow = idle_packet
            
        return output


class BaseSkill(Flow[SkillSignal, ActionSignal]):
    """Common base for skills."""
    NAME = "base"

    def run(self, input: Optional[SkillSignal]) -> ActionSignal:
        if input is None or input.flow is None or not input.flow.active:
            # print(f"[{self.NAME}] No input")
            return ActionSignal(packet=RobotAction(skill_name="idle"))
        print(f"[{self.NAME}] Running with {input.flow}")
        action = self.run_impl(input.flow)
        return ActionSignal(packet=action) if action else ActionSignal(packet=RobotAction(skill_name="idle"))

    def run_impl(self, input: SkillInput) -> Optional[RobotAction]:
        raise NotImplementedError()


class ApproachSkill(BaseSkill):
    """
    Skill: Approach a target (x,y) using a simple P-controller.
    """
    NAME = "approach"
    GAIN = 1.0
    MAX_SPEED = 2.0
    ARRIVAL_DIST = 0.1

    def run_impl(self, input: SkillInput) -> Optional[RobotAction]:
        dx = input.target_x - input.state.x
        dy = input.target_y - input.state.y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist < self.ARRIVAL_DIST:
            # Arrived
            return RobotAction(vx=0.0, vy=0.0, skill_name=self.NAME)

        # P-control
        vx = dx * self.GAIN
        vy = dy * self.GAIN

        # Clip speed
        speed = math.sqrt(vx*vx + vy*vy)
        if speed > self.MAX_SPEED:
            scale = self.MAX_SPEED / speed
            vx *= scale
            vy *= scale

        return RobotAction(vx=vx, vy=vy, skill_name=self.NAME)


class ManipulateSkill(BaseSkill):
    """
    Skill: Manipulate object (toggle gripper).
    """
    NAME = "manipulate"

    def run_impl(self, input: SkillInput) -> Optional[RobotAction]:
        # Stationary while manipulating
        action = RobotAction(vx=0.0, vy=0.0, skill_name=self.NAME)

        # Toggle logic based on current state
        if input.state.holding_object:
            # Place
            action.set_gripper_open = True
        else:
            # Pick
            action.set_gripper_open = False
            
        return action


class ActionArbiter(Flow[ArbiterInput, ActionSignal]):
    """
    Selects the active action.
    """

    def run(self, input: ArbiterInput) -> Optional[ActionSignal]:
        if input.approach_packet is not None and input.approach_packet.skill_name != "idle":
             print(f"[Arbiter] Selected APPROACH: {input.approach_packet}")
             return ActionSignal(packet=input.approach_packet)
        
        if input.manipulate_packet is not None and input.manipulate_packet.skill_name != "idle":
             print(f"[Arbiter] Selected MANIPULATE: {input.manipulate_packet}")
             return ActionSignal(packet=input.manipulate_packet)
             
        # No action
        return None


# =============================================================================
# FAN-IN VARIANTS (Simpler!)
# =============================================================================

@io
@dataclass
class ArbiterInputFanIn:
    """
    Inputs for Fan-in Arbiter.
    With Fan-in, multiple skills write to this SAME 'packet' port.
    No need for 'approach_packet', 'manipulate_packet', etc.
    """
    packet: Optional[RobotAction] = None


class ActionArbiterFanIn(Flow[ArbiterInputFanIn, ActionSignal]):
    """
    Selects the active action (Fan-in Version).
    """

    def run(self, input: ArbiterInputFanIn) -> Optional[ActionSignal]:
        # Fan-in automatically delivers the latest packet from ANY skill.
        # We just filter for idle/None.
        
        if input.packet is not None and input.packet.skill_name != "idle":
             print(f"[Arbiter] Selected {input.packet.skill_name.upper()}: {input.packet}")
             return ActionSignal(packet=input.packet)
        
        # No action or idle
        return None
