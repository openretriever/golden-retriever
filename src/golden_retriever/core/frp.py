"""
Functional Reactive Programming Primitives for Retriever Flow v5.0

This module provides the core FRP abstractions that power time-aware execution
while remaining hidden from most users through the Flow compilation interface.

Design Principles:
- Behaviors: Time-varying values that can be sampled at any time
- EventStreams: Discrete occurrences with timestamps
- Efficient implementation suitable for real-time robotics
- Thread-safe for multi-rate execution
"""

import threading
import time
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from retriever.core.rt.frp import Behavior, EventStream

T = TypeVar('T')


@dataclass
class TimestampedBuffer:
    """Thread-safe buffer for storing timestamped values."""
    max_size: int
    
    def __post_init__(self):
        self.data: List[Tuple[float, Any]] = []
        self._lock = threading.Lock()
    
    def add(self, timestamp: float, value: Any):
        """Add timestamped value to buffer."""
        with self._lock:
            self.data.append((timestamp, value))
            # Keep buffer size limited
            if len(self.data) > self.max_size:
                self.data = self.data[-self.max_size:]
    
    def get_latest_before(self, t: float) -> Optional[Any]:
        """Get most recent value before given time."""
        with self._lock:
            if not self.data:
                return None
            
            # Find latest value before time t
            for timestamp, value in reversed(self.data):
                if timestamp <= t:
                    return value
            
            # If no value before t, return oldest value
            return self.data[0][1] if self.data else None
    
    def get_all_in_window(self, current_time: float, window: float) -> List[Tuple[float, Any]]:
        """Get all values within time window."""
        with self._lock:
            cutoff_time = current_time - window
            return [(t, v) for t, v in self.data if t >= cutoff_time]
    
    def clear_old(self, current_time: float, max_age: float):
        """Remove values older than max_age."""
        with self._lock:
            cutoff_time = current_time - max_age
            self.data = [(t, v) for t, v in self.data if t >= cutoff_time]


class RateCoordinator:
    """Coordinates multiple flows operating at different rates.
    
    This handles the complexity of coordinating different component frequencies:
    - 30Hz perception data
    - 1Hz strategic planning  
    - 30Hz tactical execution
    """
    
    def __init__(self):
        self.flow_rates: Dict[str, float] = {}
        self.buffers: Dict[str, TimestampedBuffer] = {}
        self._lock = threading.Lock()
    
    def register_flow(self, flow_id: str, rate: float):
        """Register a flow with its target rate."""
        with self._lock:
            self.flow_rates[flow_id] = rate
            self.buffers[flow_id] = TimestampedBuffer(max_size=1000)
    
    def update_value(self, flow_id: str, value: Any, timestamp: float):
        """Update the latest value for a flow."""
        with self._lock:
            if flow_id in self.buffers:
                self.buffers[flow_id].add(timestamp, value)
    
    def get_value_at_time(self, flow_id: str, t: float) -> Optional[Any]:
        """Get the appropriate value for a flow at given time."""
        with self._lock:
            if flow_id not in self.buffers:
                return None
            
            buffer = self.buffers[flow_id]
            return buffer.get_latest_before(t)
    
    def create_coordinated_behavior(self, behaviors: Dict[str, Behavior]) -> Behavior:
        """Create a behavior that coordinates multiple rate-limited behaviors."""
        def coordinated_sample(t: float):
            results = {}
            for flow_id, behavior in behaviors.items():
                # Sample each behavior and update buffer
                value = behavior.at_time(t)
                self.update_value(flow_id, value, t)
                results[flow_id] = value
            return results
        
        return Behavior(coordinated_sample)
    
    def cleanup_old_data(self, current_time: float, max_age: float = 300.0):
        """Clean up old data from buffers."""
        with self._lock:
            for buffer in self.buffers.values():
                buffer.clear_old(current_time, max_age)


class EventManager:
    """Manages event streams and their coordination.
    
    Handles event-driven behaviors like obstacle detection triggering replanning.
    """
    
    def __init__(self):
        self.event_streams: Dict[str, EventStream] = {}
        self.event_handlers: Dict[str, List[Callable[[Any, float], None]]] = {}
        self._lock = threading.Lock()
    
    def register_event_stream(self, name: str, stream: EventStream):
        """Register an event stream."""
        with self._lock:
            self.event_streams[name] = stream
            if name not in self.event_handlers:
                self.event_handlers[name] = []
    
    def add_event_handler(self, event_name: str, handler: Callable[[Any, float], None]):
        """Add handler for specific event type."""
        with self._lock:
            if event_name not in self.event_handlers:
                self.event_handlers[event_name] = []
            self.event_handlers[event_name].append(handler)
    
    def process_events(self, current_time: float, window: float = 0.1):
        """Process recent events and call handlers."""
        with self._lock:
            for event_name, stream in self.event_streams.items():
                recent_events = stream.get_recent(current_time, window)
                handlers = self.event_handlers.get(event_name, [])
                
                for event_value in recent_events:
                    for handler in handlers:
                        try:
                            handler(event_value, current_time)
                        except Exception as e:
                            print(f"Error in event handler for {event_name}: {e}")
    
    def create_merged_event_stream(self, *event_names: str) -> EventStream[Tuple[str, Any]]:
        """Create event stream that merges multiple named event streams."""
        def merged_stream():
            all_events = []
            with self._lock:
                for name in event_names:
                    if name in self.event_streams:
                        stream_events = self.event_streams[name].stream()
                        # Tag events with their source name
                        tagged_events = [(t, (name, v)) for t, v in stream_events]
                        all_events.extend(tagged_events)
            
            return sorted(all_events, key=lambda x: x[0])
        
        return EventStream(merged_stream)


# Utility functions for creating behaviors and events

def constant_behavior(value: T) -> Behavior[T]:
    """Create behavior that always returns the same value."""
    return Behavior(lambda t: value)


def time_behavior() -> Behavior[float]:
    """Create behavior that returns current time."""
    return Behavior(lambda t: t)


def empty_event_stream() -> EventStream[T]:
    """Create empty event stream."""
    return EventStream(lambda: [])


def single_event(timestamp: float, value: T) -> EventStream[T]:
    """Create event stream with single event."""
    return EventStream(lambda: [(timestamp, value)])


def periodic_events(period: float, value_fn: Callable[[float], T]) -> EventStream[T]:
    """Create event stream that generates events periodically."""
    start_time = time.time()
    
    def periodic_stream():
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Generate events at period intervals
        events = []
        event_count = int(elapsed / period)
        for i in range(event_count + 1):
            event_time = start_time + (i * period)
            if event_time <= current_time:
                event_value = value_fn(event_time)
                events.append((event_time, event_value))
        
        return events
    
    return EventStream(periodic_stream)


# Advanced composition utilities

def switch_behavior(control: Behavior[bool], 
                   true_behavior: Behavior[T], 
                   false_behavior: Behavior[T]) -> Behavior[T]:
    """Switch between two behaviors based on control behavior."""
    def switched_sample(t: float) -> T:
        if control.at_time(t):
            return true_behavior.at_time(t)
        else:
            return false_behavior.at_time(t)
    
    return Behavior(switched_sample)


def until_event(behavior: Behavior[T], 
                event_stream: EventStream[Any],
                default_value: T) -> Behavior[T]:
    """Behavior that switches to default value when event occurs."""
    def until_sample(t: float) -> T:
        recent_events = event_stream.get_recent(t, window=0.001)  # Very recent
        if recent_events:
            return default_value
        else:
            return behavior.at_time(t)
    
    return Behavior(until_sample)
