"""Factory pattern for creating different types of Spot connection managers.

Provides unified interface for choosing between mock, direct SDK, and wrapper implementations
based on deployment context and requirements.
"""
from __future__ import annotations

import os
from typing import Literal, Union
from .connection import SpotConnectionConfig, MockSpotConnectionManager, SpotConnectionManager
from ..connection_base import RobotConnection

try:
    from .bdai_wrapper_connection import BdaiSpotConnectionManager
    BDAI_WRAPPER_AVAILABLE = True
except ImportError:
    BDAI_WRAPPER_AVAILABLE = False

def create_spot_manager(cfg: SpotConnectionConfig = None) -> RobotConnection:
    """Create the best Spot connection manager automatically.
    
    Uses mock for development, BDAI wrapper for real hardware.
    No choices to make - just works.
    
    Args:
        cfg: Configuration for the connection. If None, uses default config.
        
    Returns:
        RobotConnection instance (automatically selected)
    """
    if cfg is None:
        cfg = get_default_config()
    
    # Simple logic: mock for development, best implementation for hardware
    if cfg.host == "mock" or not cfg.host:
        return MockSpotConnectionManager(cfg)
    
    # Use best available implementation for real hardware
    if BDAI_WRAPPER_AVAILABLE:
        return BdaiSpotConnectionManager(cfg)
    else:
        return SpotConnectionManager(cfg)  # Fallback to direct SDK


class SpotConnectionFactory:
    """Legacy factory class - use create_spot_manager() function instead."""
    
    @staticmethod
    def _auto_select_type() -> ConnectionType:
        """Automatically select the best available connection type."""
        # Check environment variable override
        env_type = os.getenv("SPOT_CONNECTION_TYPE", "").lower()
        if env_type in ("mock", "direct_sdk", "bdai_wrapper"):
            return env_type
        
        # Check if we have real robot configuration
        host = os.getenv("SPOT_IP") or os.getenv("SPOT_HOST")
        if not host or host == "mock":
            return "mock"
        
        # Prefer wrapper if available, fall back to direct SDK
        if BDAI_WRAPPER_AVAILABLE:
            return "bdai_wrapper"
        else:
            return "direct_sdk"
    
def get_default_config() -> SpotConnectionConfig:
        """Get default configuration from environment variables."""
        return SpotConnectionConfig(
            host=os.getenv("SPOT_IP", os.getenv("SPOT_HOST", "mock")),
            username=os.getenv("SPOT_USERNAME", "admin"),
            password=os.getenv("SPOT_PASSWORD", ""),
            app_token=os.getenv("SPOT_APP_TOKEN"),
            client_name=os.getenv("SPOT_CLIENT_NAME", "retriever-spot"),
            heartbeat_sec=float(os.getenv("SPOT_HEARTBEAT_SEC", "2.0")),
            command_timeout_sec=float(os.getenv("SPOT_COMMAND_TIMEOUT_SEC", "10.0"))
        )
    
    @staticmethod
    def get_available_types() -> list[ConnectionType]:
        """Get list of available connection types based on installed dependencies."""
        types = ["mock", "direct_sdk", "auto"]
        
        if BDAI_WRAPPER_AVAILABLE:
            types.append("bdai_wrapper")
            
        return types
    
    @staticmethod
    def print_configuration_help() -> None:
        """Print help for configuring Spot connections via environment variables."""
        print("🤖 Spot Connection Configuration")
        print("=" * 40)
        print("Available connection types:", ", ".join(SpotConnectionFactory.get_available_types()))
        print("\nEnvironment Variables:")
        print("  SPOT_CONNECTION_TYPE    - Connection type (mock, direct_sdk, bdai_wrapper, auto)")
        print("  SPOT_IP / SPOT_HOST     - Robot IP address (use 'mock' for simulation)")
        print("  SPOT_USERNAME           - Robot username (default: admin)")
        print("  SPOT_PASSWORD           - Robot password")
        print("  SPOT_APP_TOKEN          - App token (alternative to username/password)")
        print("  SPOT_CLIENT_NAME        - Client identifier (default: retriever-spot)")
        print("  SPOT_HEARTBEAT_SEC      - Heartbeat interval (default: 2.0)")
        print("  SPOT_COMMAND_TIMEOUT_SEC - Command timeout (default: 10.0)")
        print("\nExample usage:")
        print("  export SPOT_CONNECTION_TYPE=mock")
        print("  export SPOT_IP=192.168.1.100")
        print("  export SPOT_USERNAME=admin")
        print("  export SPOT_PASSWORD=your_password")
        
        current_config = SpotConnectionFactory.get_default_config()
        print(f"\nCurrent configuration:")
        print(f"  Type: {SpotConnectionFactory._auto_select_type()}")
        print(f"  Host: {current_config.host}")
        print(f"  Username: {current_config.username}")
        print(f"  Client: {current_config.client_name}")
        print(f"  Heartbeat: {current_config.heartbeat_sec}s")


# Convenience functions for common usage patterns
def create_spot_manager(connection_type: ConnectionType = "auto") -> RobotConnection:
    """Create a Spot connection manager with default configuration."""
    return SpotConnectionFactory.create_manager(connection_type)


def create_mock_spot() -> RobotConnection:
    """Create a mock Spot connection for testing."""
    return SpotConnectionFactory.create_manager("mock")


def create_real_spot(use_wrapper: bool = True) -> RobotConnection:
    """Create a real Spot connection, choosing implementation based on preference."""
    connection_type = "bdai_wrapper" if use_wrapper else "direct_sdk"
    return SpotConnectionFactory.create_manager(connection_type)