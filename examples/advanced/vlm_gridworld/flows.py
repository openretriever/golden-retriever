"""
Flow definitions for VLM GridWorld Navigation.

This module defines the Retriever Flows for:
- GridEnvFlow: Environment that produces image observations
- VLMAgentFlow: VLM-based agent that processes images and outputs actions
- RerunLoggerFlow: Visualization logging to Rerun
"""

import base64
import io
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np

from retriever.flow import Flow, io
from .env import GridWorld, GridState

logger = logging.getLogger(__name__)


# ============================================================================
# Flow I/O Types
# ============================================================================

@io
@dataclass
class GridObservation:
    """Observation from the GridWorld environment."""
    image: np.ndarray              # RGB observation image
    position: Optional[tuple]      # Current (row, col) position
    goal: Optional[tuple]          # Goal (row, col) position
    step_count: Optional[int]
    reward: Optional[float]
    total_reward: Optional[float]
    done: bool
    episode: int = 0


@io
@dataclass
class AgentAction:
    """Action produced by the VLM agent."""
    action: str                    # "up", "down", "left", "right"
    reasoning: str = ""            # VLM chain-of-thought explanation
    confidence: float = 1.0        # Confidence score (0-1)


# ============================================================================
# GridEnvFlow - Environment Flow
# ============================================================================

class GridEnvFlow(Flow[AgentAction, GridObservation]):
    """
    GridWorld environment as a Retriever Flow.
    
    Receives actions, executes them, and returns image observations.
    Automatically resets when episode ends.
    """
    
    def __init__(
        self,
        size: int = 8,
        max_steps: int = 50,
        cell_size: int = 64,
    ):
        super().__init__()
        self.size = size
        self.max_steps = max_steps
        self.cell_size = cell_size
        
    def init_config(self) -> dict:
        return {
            "size": self.size,
            "max_steps": self.max_steps,
            "cell_size": self.cell_size,
        }
    
    def init(self):
        self.env = GridWorld(
            size=self.size,
            max_steps=self.max_steps,
            cell_size=self.cell_size,
        )
        self.episode = 0
        self._state = self.env.reset()
        logger.info(f"[GridEnv] Initialized {self.size}x{self.size} grid, goal={self.env.goal}")
    
    def run(self, inp: AgentAction) -> GridObservation:
        # Handle first step or reset
        if inp.action is None or self._state.done:
            if self._state.done:
                self.episode += 1
                logger.info(f"[GridEnv] Episode {self.episode} ended. Total reward: {self._state.total_reward:.2f}")
            self._state = self.env.reset()
        else:
            # Execute action
            self._state = self.env.step(inp.action)
            logger.debug(f"[GridEnv] Action: {inp.action} -> Pos: {self._state.agent_pos}")
        
        # Render observation
        image = self.env.render()
        
        return GridObservation(
            image=image,
            position=self._state.agent_pos,
            goal=self._state.goal_pos,
            step_count=self._state.step_count,
            reward=self._state.reward,
            total_reward=self._state.total_reward,
            done=self._state.done,
            episode=self.episode,
        )


# ============================================================================
# VLMAgentFlow - Vision-Language Model Agent
# ============================================================================

class VLMAgentFlow(Flow[GridObservation, AgentAction]):
    """
    VLM-based agent for GridWorld navigation.
    
    Processes image observations using a Vision-Language Model
    and outputs navigation actions with reasoning.
    
    Args:
        client: LLM client type ("openai", "gemini", or "mock")
        model: Specific model name (e.g., "gpt-4o", "gemini-1.5-flash")
        mock: If True, use heuristic policy instead of real VLM
        temperature: Sampling temperature for VLM
    """
    
    SYSTEM_PROMPT = """You are a navigation agent in a grid world. 
Your goal is to reach the green square (goal) from your current position (blue circle).

The grid shows:
- Blue circle: Your current position
- Green square: The goal you need to reach
- Gray trail: Your previous path

You can move: up, down, left, right

Analyze the image and choose the best action to reach the goal efficiently.
Respond with a single valid JSON object containing "action" and "reasoning".
Example:
{"action": "right", "reasoning": "The goal is to the east, so moving right reduces the distance."}

Ensure:
1. "action" is one of: up, down, left, right
2. "reasoning" is a short sentence explaining WHY.
3. No markdown formatting (like ```json). Just the raw JSON string."""

    def __init__(
        self,
        client: str = "gemini",
        model: Optional[str] = None,
        mock: bool = False,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.client_type = client
        self.model = model
        self.mock = mock
        self.temperature = temperature
        
    def init_config(self) -> dict:
        return {
            "client": self.client_type,
            "model": self.model,
            "mock": self.mock,
        }
    
    def init(self):
        self._client = None
        self._last_action_time = 0
        
        if not self.mock:
            self._init_client()
    
    def _init_client(self):
        """Initialize the VLM client."""
        try:
            if self.client_type == "openai":
                import openai
                self._client = openai.OpenAI()
                self.model = self.model or "gpt-4o"
                logger.info(f"[VLMAgent] Using OpenAI {self.model}")
                
            elif self.client_type == "gemini":
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not set")
                genai.configure(api_key=api_key)
                self.model = self.model or "gemini-1.5-flash"
                self._client = genai.GenerativeModel(self.model)
                logger.info(f"[VLMAgent] Using Gemini {self.model}")
                
            else:
                logger.warning(f"[VLMAgent] Unknown client '{self.client_type}', using mock")
                self.mock = True
                
        except Exception as e:
            logger.warning(f"[VLMAgent] Failed to init client: {e}. Using mock.")
            self.mock = True
    
    def run(self, obs: GridObservation) -> AgentAction:
        if obs.image is None:
            return AgentAction(action="right", reasoning="No observation yet")
        
        if obs.done:
            return AgentAction(action="", reasoning="Episode complete")
        
        if obs.position is None:
            return AgentAction(action="right", reasoning="Position not available")
        
        if self.mock:
            return self._mock_policy(obs)
        
        return self._vlm_policy(obs)
    
    def _mock_policy(self, obs: GridObservation) -> AgentAction:
        """Simple heuristic policy for testing without API."""
        row, col = obs.position
        goal_row, goal_col = obs.goal
        
        # Add some randomness for variety
        if random.random() < 0.1:
            action = random.choice(["up", "down", "left", "right"])
            return AgentAction(
                action=action,
                reasoning=f"[MOCK] Random exploration",
                confidence=0.5,
            )

        # Greedy towards goal
        if row < goal_row:
            action = "down"
            reasoning = f"[MOCK] Moving down towards goal (row {row} -> {goal_row})"
        elif row > goal_row:
            action = "up"
            reasoning = f"[MOCK] Moving up towards goal (row {row} -> {goal_row})"
        elif col < goal_col:
            action = "right"
            reasoning = f"[MOCK] Moving right towards goal (col {col} -> {goal_col})"
        elif col > goal_col:
            action = "left"
            reasoning = f"[MOCK] Moving left towards goal (col {col} -> {goal_col})"
        else:
            action = "right"
            reasoning = "[MOCK] At goal!"
        
        return AgentAction(action=action, reasoning=reasoning, confidence=0.9)
    
    def _vlm_policy(self, obs: GridObservation) -> AgentAction:
        """Query VLM for action decision."""
        try:
            # Encode image to base64
            from PIL import Image
            pil_img = Image.fromarray(obs.image)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create prompt with context
            step = obs.step_count if obs.step_count is not None else 0
            rew = obs.total_reward if obs.total_reward is not None else 0.0
            
            prompt = f"""Current position: {obs.position}
Goal position: {obs.goal}
Steps taken: {step}
Total reward: {rew:.2f}

Look at the grid image and decide the best action."""

            # Call VLM
            if self.client_type == "openai":
                response = self._call_openai(prompt, img_b64)
            else:
                response = self._call_gemini(prompt, pil_img)
            
            # Parse response
            return self._parse_response(response)
            
        except Exception as e:
            logger.error(f"[VLMAgent] VLM call failed: {e}")
            return self._mock_policy(obs)
    
    def _call_openai(self, prompt: str, img_b64: str) -> str:
        """Call OpenAI Vision API."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                    ],
                },
            ],
            temperature=self.temperature,
            max_tokens=150,
        )
        return response.choices[0].message.content
    
    def _call_gemini(self, prompt: str, pil_img) -> str:
        """Call Gemini Vision API."""
        full_prompt = f"{self.SYSTEM_PROMPT}\n\n{prompt}"
        response = self._client.generate_content([full_prompt, pil_img])
        response.resolve()
        logger.info(f"[VLMAgent] Raw Gemini Response: {response.text}") # Debug print
        return response.text
    
    def _parse_response(self, response: str) -> AgentAction:
        """Parse VLM JSON response into AgentAction."""
        try:
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
                
                action = data.get("action", "right").lower().strip()
                # Try multiple keys for reasoning
                reasoning = (
                    data.get("reasoning") or
                    data.get("explanation") or
                    data.get("thought") or
                    data.get("rationale") or
                    ""
                )
                
                if not reasoning:
                    reasoning = f"[DEBUG] Raw JSON: {json_str}" # Fallback to debug raw json
                
                if action not in ["up", "down", "left", "right"]:
                    action = "right"
                    reasoning = f"[PARSE] Invalid action, defaulting. Original: {response}"
                
                return AgentAction(action=action, reasoning=reasoning, confidence=0.8)
                
        except Exception as e:
            logger.warning(f"[VLMAgent] Parse failed: {e}. Response: {response[:100]}")
        
        return AgentAction(action="right", reasoning=f"[PARSE_ERROR] {response[:50]}", confidence=0.3)


# ============================================================================
# RerunLoggerFlow - Visualization
# ============================================================================

class RerunLoggerFlow(Flow[GridObservation, None]):
    """
    Log GridWorld state to Rerun for visualization.
    
    Displays:
    - Grid image with agent and goal
    - Scalar metrics (reward, steps)
    - Text annotations (agent reasoning)
    """
    
    def __init__(self, app_id: str = "vlm_gridworld"):
        super().__init__()
        self.app_id = app_id
        
    def init(self):
        try:
            import rerun as rr
            rr.init(self.app_id, spawn=True)
            self._rr = rr
            logger.info(f"[Rerun] Visualization started: {self.app_id}")
        except ImportError:
            logger.warning("[Rerun] rerun not installed, visualization disabled")
            self._rr = None
    
    def run(self, obs: GridObservation) -> None:
        if self._rr is None or obs.image is None:
            return
        
        rr = self._rr
        
        # Handle potential None values for safety
        step = obs.step_count if obs.step_count is not None else 0
        episode = obs.episode if obs.episode is not None else 0
        
        # Set time
        rr.set_time("step", sequence=step)
        rr.set_time("episode", sequence=episode)
        
        # Log image
        rr.log("grid/image", rr.Image(obs.image))
        
        # Log scalars
        if obs.reward is not None:
            rr.log("metrics/reward", rr.Scalars([obs.reward]))
        if obs.total_reward is not None:
            rr.log("metrics/total_reward", rr.Scalars([obs.total_reward]))
        rr.log("metrics/step", rr.Scalars([step]))
        
        # Log position
        if obs.position:
            rr.log("state/position_row", rr.Scalars([obs.position[0]]))
            rr.log("state/position_col", rr.Scalars([obs.position[1]]))
        
        # Log episode info
        if obs.done:
            total = obs.total_reward if obs.total_reward is not None else 0.0
            rr.log("events/episode_end", rr.TextLog(
                f"Episode {episode} ended with reward {total:.2f}"
            ))


# ============================================================================
# Agent Reasoning Logger (for VLM explanations)
# ============================================================================

class ReasoningLoggerFlow(Flow[AgentAction, None]):
    """Log VLM reasoning to console and Rerun."""
    
    def init(self):
        try:
            import rerun as rr
            self._rr = rr
        except ImportError:
            self._rr = None
    
    def run(self, action: AgentAction) -> None:
        if action.action:
            print(f"[VLM] Action: {action.action} | Reasoning: {action.reasoning}")

        if self._rr and action.reasoning:
            # Note: This flow doesn't receive the observation, so we can't sync purely on 'step'.
            # However, since it runs in the same pipeline iteration, Rerun's log_time (wall clock)
            # will naturally cluster these events with the observation logs.
            self._rr.log("vlm/reasoning", self._rr.TextLog(action.reasoning))
            self._rr.log("vlm/action", self._rr.TextLog(action.action))
