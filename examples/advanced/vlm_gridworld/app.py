#!/usr/bin/env python3
"""
VLM GridWorld Navigation - Main Application

A visual reinforcement learning example where a Vision-Language Model
navigates a gridworld by processing image observations.

Usage:
    # Mock mode (no API keys required)
    pixi run python -m examples.advanced.vlm_gridworld.app --mock
    
    # With Gemini VLM
    GEMINI_API_KEY=... pixi run python -m examples.advanced.vlm_gridworld.app --client gemini
    
    # With OpenAI GPT-4o
    OPENAI_API_KEY=... pixi run python -m examples.advanced.vlm_gridworld.app --client openai
"""

import argparse
import logging

from retriever.flow import Pipeline, Rate, Trigger
from retriever.flow.adapter import Latest

from .flows import GridEnvFlow, VLMAgentFlow, RerunLoggerFlow, ReasoningLoggerFlow


def setup_logging(level: str = "INFO"):
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="VLM GridWorld Navigation - Visual RL with Vision-Language Models"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock policy (no VLM API required)",
    )
    parser.add_argument(
        "--client",
        default="gemini",
        choices=["openai", "gemini"],
        help="VLM client to use (default: gemini)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Specific model name (e.g., gpt-4o, gemini-1.5-flash)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Grid size (default: 8x8)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum steps per episode (default: 50)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Total run duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--hz",
        type=float,
        default=0.5,
        help="Environment step rate in Hz (default: 0.5, i.e., 1 step per 2s)",
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
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    # Print configuration
    print("=" * 60)
    print("VLM GridWorld Navigation")
    print("=" * 60)
    print(f"  Grid Size:    {args.grid_size}x{args.grid_size}")
    print(f"  Max Steps:    {args.max_steps}")
    print(f"  Step Rate:    {args.hz} Hz ({1/args.hz:.1f}s per step)")
    print(f"  VLM Client:   {'MOCK' if args.mock else args.client}")
    print(f"  Model:        {args.model or 'default'}")
    print(f"  Duration:     {args.duration}s")
    print(f"  Rerun:        {'disabled' if args.no_rerun else 'enabled'}")
    print("=" * 60)
    
    # Create Flows
    # Environment runs at specified Hz (slow for VLM inference)
    env = GridEnvFlow(
        size=args.grid_size,
        max_steps=args.max_steps,
    ) @ Rate(hz=args.hz)
    
    # Agent triggered by new observations
    agent = VLMAgentFlow(
        client=args.client,
        model=args.model,
        mock=args.mock,
    ) @ Trigger("image")
    
    # Build Pipeline
    pipe = Pipeline("vlm_gridworld")
    
    # Closed-loop connection
    pipe.connect(env, agent)  # Env -> Agent (observations)
    pipe.connect(agent, env, sync=Latest())  # Agent -> Env (actions)
    
    # Optional visualization
    if not args.no_rerun:
        logger = RerunLoggerFlow() @ Trigger("image")
        pipe.connect(env, logger)
        
        reasoning_logger = ReasoningLoggerFlow() @ Trigger("action")
        pipe.connect(agent, reasoning_logger)
    
    print("\nStarting VLM GridWorld...")
    if args.mock:
        print("[Note] Running in MOCK mode - using heuristic policy")
    else:
        print(f"[Note] Using {args.client} VLM - ensure API key is set")
    
    if not args.no_rerun:
        print("[Note] Check Rerun viewer for visualization")
    
    print()
    
    # Run
    pipe.run(backend=args.backend, duration=args.duration, blocking=True)
    
    print("\n[Done] VLM GridWorld completed.")


if __name__ == "__main__":
    main()
