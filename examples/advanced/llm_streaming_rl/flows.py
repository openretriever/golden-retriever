"""
Flow definitions for LLM Streaming RL.

This module defines the Retriever Flows for:
- TextEnvFlow: Text-based 20 Questions environment
- LLMAgentFlow: LLM agent with streaming support
- StreamMonitorFlow: Real-time token stream visualization
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from retriever.flow import Flow, io
from .env import TwentyQuestionsEnv, EnvState
from .streaming import (
    StreamingLLMClient,
    MockStreamingClient,
    StreamChunk,
    StreamingConfig,
    create_streaming_client,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Flow I/O Types
# ============================================================================

@io
@dataclass
class TextObservation:
    """Observation from text environment."""
    text: str                          # Current observation text
    history: List[Tuple[str, str]]     # [(question, answer), ...]
    step_count: Optional[int]
    reward: Optional[float]
    total_reward: Optional[float]
    done: bool
    episode: int = 0
    info: dict = field(default_factory=dict)


@io
@dataclass
class LLMAction:
    """Action produced by LLM agent."""
    action: str                        # The action/question to take
    full_response: str = ""            # Complete LLM response
    reasoning: str = ""                # Extracted reasoning (if any)
    is_guess: bool = False             # True if this is a final guess
    stream_chunks: List[str] = field(default_factory=list)  # Token chunks for logging


@io
@dataclass
class StreamEvent:
    """A streaming token event for real-time display."""
    tokens: str
    is_complete: bool
    cumulative: str
    chunk_index: int
    timestamp: float


# ============================================================================
# TextEnvFlow - 20 Questions Environment
# ============================================================================

class TextEnvFlow(Flow[LLMAction, TextObservation]):
    """
    20 Questions environment as a Retriever Flow.
    
    Receives questions/guesses from LLM agent and returns text observations.
    """
    
    def __init__(
        self,
        max_questions: int = 20,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.max_questions = max_questions
        self.seed = seed
        
    def init_config(self) -> dict:
        return {
            "max_questions": self.max_questions,
            "seed": self.seed,
        }
    
    def init(self):
        self.env = TwentyQuestionsEnv(
            max_questions=self.max_questions,
            seed=self.seed,
        )
        self.episode = 0
        self._state = self.env.reset()
        logger.info(f"[TextEnv] Initialized 20 Questions (max={self.max_questions})")
        logger.info(f"[TextEnv] Secret word (debug): {self.env.secret}")
    
    def run(self, inp: LLMAction) -> TextObservation:
        # Handle first step or episode reset
        if inp.action is None or inp.action == "" or self._state.done:
            if self._state.done:
                self.episode += 1
                logger.info(f"[TextEnv] Episode {self.episode} ended. Score: {self._state.total_reward:.2f}")
                logger.info(f"[TextEnv] New game starting...")
            self._state = self.env.reset()
            logger.info(f"[TextEnv] Secret word (debug): {self.env.secret}")
        else:
            # Process action
            self._state = self.env.step(inp.action)
            logger.debug(f"[TextEnv] Action: {inp.action[:50]}... -> Reward: {self._state.reward:+.2f}")
        
        return TextObservation(
            text=self._state.observation,
            history=self._state.history,
            step_count=self._state.step_count,
            reward=self._state.reward,
            total_reward=self._state.total_reward,
            done=self._state.done,
            episode=self.episode,
            info=self._state.info,
        )


# ============================================================================
# LLMAgentFlow - Streaming LLM Agent
# ============================================================================

class LLMAgentFlow(Flow[TextObservation, LLMAction]):
    """
    LLM-based agent for 20 Questions with streaming support.
    
    This flow demonstrates:
    - Streaming token generation from LLM APIs
    - Buffered chunk emission for smooth output
    - Integration with Retriever's reactive dataflow
    
    Args:
        model: Model name (e.g., "gpt-4o", "gpt-4o-mini")
        mock: Use mock streaming client (no API required)
        stream: Enable streaming output (vs batch)
        chunk_size: Tokens per chunk when streaming
    """
    
    SYSTEM_PROMPT = """You are playing 20 Questions. The host is thinking of a word and you must guess it.

Rules:
- Ask strategic yes/no questions to narrow down possibilities
- Use "ask: <question>" to ask a question
- Use "guess: <word>" when you're confident about the answer
- Think systematically: start broad (animal? object? place?) then narrow down

Strategy tips:
- Binary search: divide the possibility space in half each question
- Track what you've learned from previous answers
- Don't waste questions on unlikely guesses

Respond with ONLY your action (ask: or guess:), followed by brief reasoning.
Format: 
ask: Is it alive?
(Reasoning: Starting broad to determine if it's living or non-living)"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        mock: bool = False,
        stream: bool = True,
        chunk_size: int = 3,
        temperature: float = 0.7,
    ):
        super().__init__()
        self.model = model
        self.mock = mock
        self.stream_enabled = stream
        self.chunk_size = chunk_size
        self.temperature = temperature
        
    def init_config(self) -> dict:
        return {
            "model": self.model,
            "mock": self.mock,
            "stream": self.stream_enabled,
        }
    
    def init(self):
        self._client = create_streaming_client(
            mock=self.mock,
            model=self.model,
            temperature=self.temperature,
            delay_ms=30 if self.mock else 0,
        )
        self._stream_config = StreamingConfig(
            chunk_size=self.chunk_size,
            emit_on_newline=True,
            emit_on_punctuation=True,
        )
        
        # For emitting stream events
        self._last_stream_chunks: List[str] = []
        
        mode = "MOCK" if self.mock else f"{self.model} (streaming={'on' if self.stream_enabled else 'off'})"
        logger.info(f"[LLMAgent] Initialized: {mode}")
    
    def run(self, obs: TextObservation) -> LLMAction:
        # Handle None/uninitialized observations
        if obs.text is None:
            return LLMAction(action="", full_response="Waiting for observation")
        
        if obs.done:
            return LLMAction(action="", full_response="Episode complete")
        
        # Build prompt with context
        prompt = self._build_prompt(obs)
        
        # Generate response
        if self.stream_enabled:
            response, chunks = self._streaming_inference(prompt)
        else:
            response = self._batch_inference(prompt)
            chunks = [response]
        
        # Parse action from response
        action, reasoning, is_guess = self._parse_response(response)
        
        logger.info(f"[LLMAgent] {'GUESS' if is_guess else 'ASK'}: {action}")
        if reasoning:
            logger.debug(f"[LLMAgent] Reasoning: {reasoning}")
        
        return LLMAction(
            action=action,
            full_response=response,
            reasoning=reasoning,
            is_guess=is_guess,
            stream_chunks=chunks,
        )
    
    def _build_prompt(self, obs: TextObservation) -> str:
        """Build prompt with game history."""
        lines = [obs.text, "\n"]
        
        if obs.history:
            lines.append("Previous Q&A:")
            for q, a in obs.history[-5:]:  # Last 5 Q&A pairs
                lines.append(f"  Q: {q}")
                lines.append(f"  A: {a}")
            lines.append("")
        
        step_count = obs.step_count or 0
        remaining = obs.info.get('questions_remaining') if obs.info else None
        total_q = remaining + step_count if remaining is not None else 20
        lines.append(f"Questions used: {step_count}/{total_q}")
        lines.append("\nWhat is your next move?")
        
        return "\n".join(lines)
    
    def _streaming_inference(self, prompt: str) -> Tuple[str, List[str]]:
        """Generate response with streaming."""
        chunks = []
        full_response = ""
        
        print("\n[LLM Streaming] ", end="", flush=True)
        
        for chunk in self._client.stream_chunks(prompt, self.SYSTEM_PROMPT, self._stream_config):
            chunks.append(chunk.tokens)
            full_response = chunk.cumulative
            
            # Print tokens as they arrive
            print(chunk.tokens, end="", flush=True)
        
        print()  # Newline after streaming
        
        self._last_stream_chunks = chunks
        return full_response, chunks
    
    def _batch_inference(self, prompt: str) -> str:
        """Generate response without streaming."""
        return self._client.complete(prompt, self.SYSTEM_PROMPT)
    
    def _parse_response(self, response: str) -> Tuple[str, str, bool]:
        """
        Parse LLM response to extract action.
        
        Returns:
            (action_string, reasoning, is_guess)
        """
        response = response.strip()
        lines = response.split("\n")
        
        action = ""
        reasoning = ""
        is_guess = False
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if line_lower.startswith("guess:"):
                action = "guess: " + line[6:].strip()
                is_guess = True
                break
            elif line_lower.startswith("ask:"):
                action = "ask: " + line[4:].strip()
                break
            elif "?" in line and not action:
                # Treat as question
                action = "ask: " + line.strip()
        
        # Extract reasoning (anything in parentheses or after the action)
        if "(" in response and ")" in response:
            start = response.find("(")
            end = response.rfind(")") + 1
            reasoning = response[start:end]
        
        # Default action if parsing failed
        if not action:
            # Try to use first line as question
            action = "ask: " + lines[0].strip() if lines else "ask: Is it alive?"
        
        return action, reasoning, is_guess


# ============================================================================
# StreamMonitorFlow - Real-time Token Display
# ============================================================================

class StreamMonitorFlow(Flow[LLMAction, None]):
    """
    Monitor and display streaming tokens from LLM.
    
    Logs to console and optionally Rerun for visualization.
    """
    
    def __init__(self, use_rerun: bool = True):
        super().__init__()
        self.use_rerun = use_rerun
        
    def init(self):
        self._rr = None
        if self.use_rerun:
            try:
                import rerun as rr
                self._rr = rr
            except ImportError:
                logger.warning("[StreamMonitor] Rerun not available")
    
    def run(self, action: LLMAction) -> None:
        if not action.stream_chunks:
            return
        
        # Log to Rerun
        if self._rr:
            rr = self._rr
            
            # Log full response
            rr.log("llm/response", rr.TextLog(action.full_response))
            
            # Log action
            rr.log("llm/action", rr.TextLog(action.action))
            
            # Log stream info
            # Log stream info
            rr.log("llm/chunks", rr.Scalars([len(action.stream_chunks)]))


# ============================================================================
# GameLoggerFlow - Episode Logging
# ============================================================================

class GameLoggerFlow(Flow[TextObservation, None]):
    """Log game state to console and Rerun."""
    
    def __init__(self, use_rerun: bool = True):
        super().__init__()
        self.use_rerun = use_rerun
        
    def init(self):
        self._rr = None
        if self.use_rerun:
            try:
                import rerun as rr
                rr.init("llm_20questions", spawn=True)
                self._rr = rr
                logger.info("[GameLogger] Rerun visualization started")
            except ImportError:
                logger.warning("[GameLogger] Rerun not available")
    
    def run(self, obs: TextObservation) -> None:
        # Handle None observations
        if obs.text is None:
            return
        
        # Console output
        step_count = obs.step_count or 0
        reward = obs.reward if obs.reward is not None else 0.0
        total_reward = obs.total_reward if obs.total_reward is not None else 0.0
        
        print("\n" + "=" * 60)
        print(obs.text)
        print(f"\n[Step: {step_count} | Reward: {reward:+.2f} | Total: {total_reward:.2f}]")
        print("=" * 60)
        
        # Rerun logging
        if self._rr:
            rr = self._rr
            
            step = obs.step_count if obs.step_count is not None else 0
            episode = obs.episode if obs.episode is not None else 0
            
            rr.set_time("step", sequence=step)
            rr.set_time("episode", sequence=episode)
            
            rr.log("game/observation", rr.TextLog(obs.text))
            
            if obs.reward is not None:
                rr.log("metrics/reward", rr.Scalars([obs.reward]))
            if obs.total_reward is not None:
                rr.log("metrics/total_reward", rr.Scalars([obs.total_reward]))
            rr.log("metrics/step", rr.Scalars([step]))
            
            if obs.done:
                total_rew = obs.total_reward if obs.total_reward is not None else 0.0
                rr.log("events/episode_end", rr.TextLog(
                    f"Episode {episode} ended with score {total_rew:.2f}"
                ))
