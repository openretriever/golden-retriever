"""Simple Spot connection - automatically uses best implementation.

No choices, no configuration complexity - just works.
"""
from __future__ import annotations

import os
from .connection import SpotConnectionConfig, MockSpotConnectionManager, SpotConnectionManager
from ..connection_base import RobotConnection

try:
    from .bdai_wrapper_connection import BdaiSpotConnectionManager
    BDAI_WRAPPER_AVAILABLE = True
except ImportError:
    BDAI_WRAPPER_AVAILABLE = False


def create_spot_manager() -> RobotConnection:
    """Create the best Spot connection automatically.
    
    - Mock for development (no SPOT_IP or SPOT_IP=mock)
    - BDAI wrapper for real hardware (best implementation)
    - Direct SDK as fallback
    
    No configuration needed for development.
    For hardware, just set: SPOT_IP, SPOT_USERNAME, SPOT_PASSWORD
    """
    cfg = SpotConnectionConfig(
        host=os.getenv("SPOT_IP", "mock"),
        username=os.getenv("SPOT_USERNAME", "admin"),
        password=os.getenv("SPOT_PASSWORD", ""),
        app_token=os.getenv("SPOT_APP_TOKEN"),
    )
    
    # Use mock for development
    if cfg.host == "mock":
        return MockSpotConnectionManager(cfg)
    
    # Use best implementation for real hardware
    if BDAI_WRAPPER_AVAILABLE:
        return BdaiSpotConnectionManager(cfg)
    else:
        return SpotConnectionManager(cfg)


# Legacy support
def create_mock_spot() -> RobotConnection:
    """Create mock connection for testing."""
    return MockSpotConnectionManager(SpotConnectionConfig(host="mock"))