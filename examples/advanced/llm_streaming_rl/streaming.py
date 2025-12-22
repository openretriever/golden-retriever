"""
Streaming LLM Utilities.

This module provides utilities for integrating streaming LLM responses
with Retriever's reactive flow system.

Key concepts:
- StreamingLLMClient: Wraps OpenAI streaming API
- Token buffering and chunked emission
- Integration with EventStream pattern
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Callable, Any
from queue import Queue
from threading import Thread

logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """A chunk of streamed tokens from an LLM."""
    tokens: str                    # The token(s) in this chunk
    timestamp: float               # When this chunk was received
    is_complete: bool = False      # True if this is the final chunk
    cumulative: str = ""           # All tokens received so far
    chunk_index: int = 0           # Index of this chunk


@dataclass
class StreamingConfig:
    """Configuration for streaming behavior."""
    chunk_size: int = 5            # Number of tokens per chunk (0 = token-by-token)
    emit_on_newline: bool = True   # Also emit on newlines
    emit_on_punctuation: bool = True  # Emit on sentence-ending punctuation
    max_buffer_time_ms: int = 500  # Max time to buffer before forced emit


class StreamingLLMClient:
    """
    A wrapper for OpenAI's streaming API.
    
    Provides both raw token streaming and chunked emission modes.
    
    Example:
        client = StreamingLLMClient(model="gpt-4o")
        
        # Token-by-token
        for token in client.stream("Tell me a joke"):
            print(token, end="", flush=True)
        
        # Chunked
        for chunk in client.stream_chunks("Tell me a joke"):
            print(f"[Chunk] {chunk.tokens}")
    """
    
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Initialize the streaming client.
        
        Args:
            model: OpenAI model name
            api_key: API key (or read from OPENAI_API_KEY)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Defer import and initialization
        self._client = None
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        
    def _ensure_client(self):
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai package required for streaming. Install with: pip install openai")
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream tokens one at a time.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Yields:
            Individual tokens as they arrive
        """
        self._ensure_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"[ERROR: {e}]"
    
    def stream_chunks(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[StreamingConfig] = None,
    ) -> Generator[StreamChunk, None, None]:
        """
        Stream tokens in chunks (better for UI/logging).
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            config: Chunking configuration
            
        Yields:
            StreamChunk objects with buffered tokens
        """
        config = config or StreamingConfig()
        
        buffer = []
        cumulative = []
        chunk_index = 0
        last_emit_time = time.time()
        
        def should_emit(token: str) -> bool:
            """Check if we should emit a chunk."""
            if len(buffer) >= config.chunk_size > 0:
                return True
            if config.emit_on_newline and "\n" in token:
                return True
            if config.emit_on_punctuation and any(p in token for p in ".!?"):
                return True
            if (time.time() - last_emit_time) * 1000 > config.max_buffer_time_ms:
                return True
            return False
        
        for token in self.stream(prompt, system_prompt):
            buffer.append(token)
            cumulative.append(token)
            
            if should_emit(token):
                chunk_text = "".join(buffer)
                yield StreamChunk(
                    tokens=chunk_text,
                    timestamp=time.time(),
                    is_complete=False,
                    cumulative="".join(cumulative),
                    chunk_index=chunk_index,
                )
                buffer = []
                chunk_index += 1
                last_emit_time = time.time()
        
        # Emit remaining buffer
        if buffer:
            chunk_text = "".join(buffer)
            yield StreamChunk(
                tokens=chunk_text,
                timestamp=time.time(),
                is_complete=True,
                cumulative="".join(cumulative),
                chunk_index=chunk_index,
            )
        else:
            # Mark last chunk as complete
            yield StreamChunk(
                tokens="",
                timestamp=time.time(),
                is_complete=True,
                cumulative="".join(cumulative),
                chunk_index=chunk_index,
            )
    
    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Non-streaming completion (for comparison/fallback).
        """
        self._ensure_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
        )
        
        return response.choices[0].message.content


class MockStreamingClient:
    """
    A mock streaming client for testing without API keys.
    
    Simulates streaming by yielding characters with small delays.
    """
    
    RESPONSES = {
        "greeting": "Hello! I'm a simulated LLM response. How can I help you today?",
        "question": "That's an interesting question! Let me think about it...",
        "default": "I understand. Here's my response to your query.",
    }
    
    def __init__(self, delay_ms: int = 50):
        """
        Args:
            delay_ms: Milliseconds between tokens (simulates streaming)
        """
        self.delay_ms = delay_ms
    
    def stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Simulate token streaming."""
        # Pick response based on prompt
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            response = self.RESPONSES["greeting"]
        elif "?" in prompt:
            response = self.RESPONSES["question"]
        else:
            response = self.RESPONSES["default"]
        
        # Add some context-aware content
        response += f"\n\n[This is a mock response to: '{prompt[:30]}...']"
        
        # Stream word by word
        words = response.split()
        for i, word in enumerate(words):
            time.sleep(self.delay_ms / 1000)
            yield word + (" " if i < len(words) - 1 else "")
    
    def stream_chunks(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[StreamingConfig] = None,
    ) -> Generator[StreamChunk, None, None]:
        """Simulate chunked streaming."""
        config = config or StreamingConfig()
        
        buffer = []
        cumulative = []
        chunk_index = 0
        
        for token in self.stream(prompt, system_prompt):
            buffer.append(token)
            cumulative.append(token)
            
            if len(buffer) >= max(1, config.chunk_size):
                yield StreamChunk(
                    tokens="".join(buffer),
                    timestamp=time.time(),
                    is_complete=False,
                    cumulative="".join(cumulative),
                    chunk_index=chunk_index,
                )
                buffer = []
                chunk_index += 1
        
        # Final chunk
        if buffer:
            yield StreamChunk(
                tokens="".join(buffer),
                timestamp=time.time(),
                is_complete=True,
                cumulative="".join(cumulative),
                chunk_index=chunk_index,
            )
        else:
            yield StreamChunk(
                tokens="",
                timestamp=time.time(),
                is_complete=True,
                cumulative="".join(cumulative),
                chunk_index=chunk_index,
            )
    
    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Non-streaming completion."""
        tokens = list(self.stream(prompt, system_prompt))
        return "".join(tokens)


def create_streaming_client(mock: bool = False, **kwargs) -> StreamingLLMClient | MockStreamingClient:
    """
    Factory function to create a streaming client.
    
    Args:
        mock: If True, return mock client
        **kwargs: Passed to StreamingLLMClient
        
    Returns:
        Streaming client instance
    """
    if mock:
        return MockStreamingClient(delay_ms=kwargs.get("delay_ms", 30))
    # Remove mock-only args
    kwargs.pop("delay_ms", None)
    return StreamingLLMClient(**kwargs)


# ============================================================================
# Integration with Retriever EventStream
# ============================================================================

def stream_to_event_buffer(
    client: StreamingLLMClient | MockStreamingClient,
    prompt: str,
    system_prompt: Optional[str] = None,
) -> List[tuple]:
    """
    Convert a streaming LLM call to an EventBuffer.
    
    This demonstrates bridging streaming APIs to Retriever's FRP model.
    
    Returns:
        List of (timestamp, StreamChunk) tuples
    """
    buffer = []
    for chunk in client.stream_chunks(prompt, system_prompt):
        buffer.append((chunk.timestamp, chunk))
    return buffer


if __name__ == "__main__":
    # Demo
    print("=== Mock Streaming Demo ===\n")
    
    client = MockStreamingClient(delay_ms=30)
    
    print("Token-by-token:")
    for token in client.stream("Hello, how are you?"):
        print(token, end="", flush=True)
    print("\n")
    
    print("Chunked (5 tokens per chunk):")
    for chunk in client.stream_chunks("Tell me about AI", config=StreamingConfig(chunk_size=5)):
        print(f"[Chunk {chunk.chunk_index}] {chunk.tokens!r}")
    print()
    
    print("=== OpenAI Streaming (if API key available) ===\n")
    if os.getenv("OPENAI_API_KEY"):
        client = StreamingLLMClient(model="gpt-4o-mini")
        print("Streaming from GPT-4o-mini:")
        for token in client.stream("Say 'Hello World' in three different languages"):
            print(token, end="", flush=True)
        print()
    else:
        print("(Skipped - OPENAI_API_KEY not set)")
