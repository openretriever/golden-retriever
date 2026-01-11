# VLM-Compatible Belief Updater Flow
# Simplified belief management for VLM-based planning

from typing import Optional
from dataclasses import dataclass

from retriever.flow import Flow, flow_io
from retriever.types.symbolic import State

from ..types.belief import BeliefState


@flow_io
@dataclass
class VLMBeliefInput:
    """Input for VLM belief updater."""
    observation: Optional[State] = None
    raw_observation: Optional[dict] = None  # Frame data from VLMPerceptionFlow


@flow_io
@dataclass
class VLMBeliefOutput:
    """Output for VLM belief updater."""
    belief: BeliefState = None  # Will be initialized with empty BeliefState if None
    
    def __post_init__(self):
        if self.belief is None:
            self.belief = BeliefState(data={})


class VLMBeliefUpdaterFlow(Flow[VLMBeliefInput, VLMBeliefOutput]):
    """VLM-compatible belief updater that maintains frame-based state.
    
    Simplified version that focuses on passing raw observation 
    (webcam frames) to downstream VLM flows.
    """
    
    def __init__(self, name: str = "VLMBeliefUpdater"):
        self.name = name
        self.current_belief: Optional[BeliefState] = None
        
    def init(self):
        print(f"[{self.name}] Initialized VLM belief updater")
    
    def step(self, inp: VLMBeliefInput) -> VLMBeliefOutput:
        obs = inp.observation
        raw_obs = inp.raw_observation
        
        # If no observation, return previous belief
        if obs is None and raw_obs is None:
            if self.current_belief:
                return VLMBeliefOutput(belief=self.current_belief)
            return VLMBeliefOutput(belief=None)
        
        # Build belief from observation
        belief_data = obs.data.copy() if obs else {}
        
        # Create belief state with raw observation for VLM access
        new_belief = BeliefState(
            data=belief_data,
            visual_atoms={},
            epistemic=None,
            action_history=[],
            raw_observation=raw_obs,  # Critical: VLM flows need this
        )
        
        # Log to Rerun
        rr = self.rr
        if rr:
            if raw_obs and raw_obs.get("frame"):
                frame_size = len(raw_obs["frame"])
                rr.log("belief/status", rr.TextLog(f"Updated with frame ({frame_size} bytes)"))
            else:
                rr.log("belief/status", rr.TextLog("Updated (no frame)"))
        
        self.current_belief = new_belief
        return VLMBeliefOutput(belief=new_belief)
