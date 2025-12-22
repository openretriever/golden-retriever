#!/usr/bin/env python3
"""
LLM Streaming RL - Main Application

A text-based reinforcement learning example demonstrating LLM streaming
integration with Retriever's reactive flow system.

The agent plays "20 Questions" - asking strategic yes/no questions to
guess a secret word, with streaming token display.

Usage:
    # Mock mode (no API keys required)
    pixi run python -m examples.advanced.llm_streaming_rl.app --mock
    
    # With OpenAI (streaming)
    OPENAI_API_KEY=... pixi run python -m examples.advanced.llm_streaming_rl.app
    
    # Batch mode (no streaming)
    OPENAI_API_KEY=... pixi run python -m examples.advanced.llm_streaming_rl.app --no-stream
"""

import argparse
import logging

from retriever.flow import Pipeline, Rate, Trigger
from retriever.flow.adapter import Latest

from .flows import TextEnvFlow, LLMAgentFlow, StreamMonitorFlow, GameLoggerFlow


def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="LLM Streaming RL - 20 Questions with Streaming LLM"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM (no API required)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming (use batch inference)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=15,
        help="Maximum questions per game (default: 15)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of episodes to play (default: 2)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=0.1,
        help="Turn rate in Hz (default: 0.1, i.e., 1 turn per 10s)",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        help="Disable Rerun visualization",
    )
    parser.add_argument(
        "--backend",
        default="multiprocessing",
        choices=["multiprocessing", "dora"],
        help="Execution backend (default: multiprocessing)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Calculate duration based on episodes and questions
    # Assume ~10s per question + some buffer
    estimated_duration = args.episodes * args.max_questions * (1 / args.hz) * 1.2
    
    # Print configuration
    print("=" * 60)
    print("LLM Streaming RL - 20 Questions")
    print("=" * 60)
    print(f"  Model:          {'MOCK' if args.mock else args.model}")
    print(f"  Streaming:      {'disabled' if args.no_stream else 'enabled'}")
    print(f"  Max Questions:  {args.max_questions}")
    print(f"  Episodes:       {args.episodes}")
    print(f"  Turn Rate:      {args.hz} Hz ({1/args.hz:.1f}s per turn)")
    print(f"  Est. Duration:  {estimated_duration:.0f}s")
    print(f"  Rerun:          {'disabled' if args.no_rerun else 'enabled'}")
    print("=" * 60)
    
    # Create Flows
    # Environment runs at specified Hz (slow for LLM inference)
    env = TextEnvFlow(
        max_questions=args.max_questions,
        seed=args.seed,
    ) @ Rate(hz=args.hz)
    
    # Agent triggered by new observations
    agent = LLMAgentFlow(
        model=args.model,
        mock=args.mock,
        stream=not args.no_stream,
    ) @ Trigger("text")
    
    # Build Pipeline
    pipe = Pipeline("llm_streaming_rl")
    
    # Closed-loop connection
    pipe.connect(env, agent)  # Env -> Agent (observations)
    pipe.connect(agent, env, sync=Latest())  # Agent -> Env (actions)
    
    # Visualization and monitoring
    if not args.no_rerun:
        game_logger = GameLoggerFlow(use_rerun=True) @ Trigger("text")
        pipe.connect(env, game_logger)
        
        stream_monitor = StreamMonitorFlow(use_rerun=True) @ Trigger("action")
        pipe.connect(agent, stream_monitor)
    else:
        # Minimal console logging
        game_logger = GameLoggerFlow(use_rerun=False) @ Trigger("text")
        pipe.connect(env, game_logger)
    
    print("\n🎮 Starting 20 Questions Game!")
    print("Watch the LLM ask questions and try to guess the secret word.\n")
    
    if args.mock:
        print("[Note] Running in MOCK mode - using simulated LLM responses")
    else:
        print(f"[Note] Using {args.model} - ensure OPENAI_API_KEY is set")
    
    if not args.no_stream:
        print("[Note] Streaming enabled - tokens will appear progressively")
    
    if not args.no_rerun:
        print("[Note] Check Rerun viewer for visualization")
    
    print()
    
    # Run
    pipe.run(backend=args.backend, duration=estimated_duration, blocking=True)
    
    print("\n[Done] 20 Questions game completed.")


if __name__ == "__main__":
    main()
