"""Enhanced Spot connection using bdai_spot_wrapper for comprehensive robot control."""
from __future__ import annotations

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

try:
    # Import bdai_spot_wrapper
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../external/bdai_spot_wrapper'))
    from spot_wrapper.wrapper import SpotWrapper  # type: ignore
    from spot_wrapper.wrapper_helpers import RobotState  # type: ignore
    BDAI_WRAPPER_AVAILABLE = True
except ImportError:
    BDAI_WRAPPER_AVAILABLE = False

from .connection import SpotConnectionConfig, SpotConnectionError
from ..connection_base import RobotConnection


class BdaiSpotConnectionManager(RobotConnection):
    """Enhanced Spot connection using bdai_spot_wrapper for comprehensive robot control.
    
    Provides higher-level abstractions and more reliable hardware integration
    compared to direct Boston Dynamics SDK usage.
    """

    def __init__(self, cfg: SpotConnectionConfig):
        if not BDAI_WRAPPER_AVAILABLE:
            raise SpotConnectionError(
                "bdai_spot_wrapper not available. Please install or use SpotConnectionManager."
            )
        
        self.cfg = cfg
        self._wrapper: Optional[SpotWrapper] = None
        self._last_heartbeat = 0.0
        self._busy = False
        self._estop_engaged = False

    def connect(self) -> None:
        """Initialize connection using bdai_spot_wrapper."""
        if self._wrapper is not None:
            return
        
        try:
            # Initialize SpotWrapper with configuration
            self._wrapper = SpotWrapper(
                username=self.cfg.username or "admin",
                password=self.cfg.password or "", 
                hostname=self.cfg.host,
                robot_name=self.cfg.client_name,
                logger=None,  # Use default logger
                start_estop=True,  # Enable E-Stop endpoint
                estop_timeout=9.0,
                rates={}  # Use defaults
            )
            
            # Authenticate and initialize
            self._wrapper.authenticate()
            self._wrapper.updateTasks()
            
            # Power on and stand up if not already
            if not self._wrapper.is_powered_on():
                print("Powering on robot...")
                self._wrapper.power_on()
                time.sleep(2)  # Wait for power up
            
            if not self._wrapper.is_standing():
                print("Standing robot...")
                self._wrapper.stand()
                time.sleep(2)  # Wait for standing
            
            self._last_heartbeat = time.time()
            print(f"✅ Connected to Spot at {self.cfg.host}")
            
        except Exception as e:
            raise SpotConnectionError(f"Failed to connect to Spot: {e}")

    def ensure_connected(self) -> None:
        """Ensure connection is active with heartbeat checking."""
        if self._wrapper is None:
            self.connect()
            return
        
        # Heartbeat check
        now = time.time()
        if now - self._last_heartbeat >= self.cfg.heartbeat_sec:
            try:
                # Update robot state as heartbeat
                self._wrapper.updateTasks()
                self._last_heartbeat = now
            except Exception as e:
                print(f"⚠️ Heartbeat failed, reconnecting: {e}")
                self._wrapper = None
                self.connect()

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute high-level command using bdai_spot_wrapper capabilities."""
        self.ensure_connected()
        
        if self._busy:
            return {"ok": False, "error": "Busy", "retry_after_sec": self.cfg.heartbeat_sec}
        
        typ = command.get("type")
        params = command.get("parameters", {})
        
        try:
            self._busy = True
            
            if typ == "move_to":
                return self._execute_move_to(params)
            elif typ == "move_velocity":
                return self._execute_move_velocity(params)
            elif typ == "navigate_to":
                return self._execute_navigate_to(params)
            elif typ == "stand":
                return self._execute_stand()
            elif typ == "sit":
                return self._execute_sit()
            elif typ == "stop":
                return self._execute_stop()
            elif typ == "power_on":
                return self._execute_power_on()
            elif typ == "power_off":
                return self._execute_power_off()
            elif typ in ("estop", "e_stop"):
                return self._execute_estop()
            elif typ == "estop_release":
                return self._execute_estop_release()
            elif typ == "open_gripper":
                return self._execute_open_gripper()
            elif typ == "close_gripper":
                return self._execute_close_gripper()
            elif typ == "arm_stow":
                return self._execute_arm_stow()
            elif typ == "arm_ready":
                return self._execute_arm_ready()
            elif typ == "take_picture":
                return self._execute_take_picture(params)
            elif typ == "dock":
                return self._execute_dock(params)
            else:
                return {"ok": False, "error": f"Unknown command type: {typ}"}
                
        except Exception as e:
            return {"ok": False, "error": f"Command execution failed: {e}"}
        finally:
            self._busy = False

    def _execute_move_to(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute relative movement command."""
        x = float(params.get("x", 0.0))
        y = float(params.get("y", 0.0))
        yaw = float(params.get("yaw", 0.0))
        
        # Use wrapper's trajectory command
        success = self._wrapper.trajectory_cmd(
            goal_x=x, goal_y=y, goal_heading=yaw,
            duration=float(params.get("duration", 5.0))
        )
        
        if success:
            return {"ok": True, "telemetry": {"x": x, "y": y, "yaw": yaw, "method": "trajectory"}}
        else:
            return {"ok": False, "error": "Trajectory command failed"}

    def _execute_move_velocity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute velocity command."""
        v_x = float(params.get("x", 0.0))
        v_y = float(params.get("y", 0.0))
        v_rot = float(params.get("yaw", 0.0))
        
        # Use wrapper's velocity command
        self._wrapper.velocity_cmd(v_x=v_x, v_y=v_y, v_rot=v_rot)
        
        return {"ok": True, "telemetry": {"v_x": v_x, "v_y": v_y, "v_rot": v_rot}}

    def _execute_navigate_to(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute navigation to waypoint (if graph nav is available)."""
        waypoint_id = params.get("waypoint_id")
        
        if not waypoint_id:
            return {"ok": False, "error": "waypoint_id required for navigation"}
        
        try:
            # Use wrapper's navigation capabilities
            success = self._wrapper.navigate_to_waypoint(waypoint_id)
            if success:
                return {"ok": True, "telemetry": {"waypoint_id": waypoint_id}}
            else:
                return {"ok": False, "error": f"Navigation to {waypoint_id} failed"}
        except Exception as e:
            return {"ok": False, "error": f"Navigation not available: {e}"}

    def _execute_stand(self) -> Dict[str, Any]:
        """Stand the robot."""
        try:
            self._wrapper.stand()
            return {"ok": True, "telemetry": {"standing": True}}
        except Exception as e:
            return {"ok": False, "error": f"Stand failed: {e}"}

    def _execute_sit(self) -> Dict[str, Any]:
        """Sit the robot."""
        try:
            self._wrapper.sit()
            return {"ok": True, "telemetry": {"sitting": True}}
        except Exception as e:
            return {"ok": False, "error": f"Sit failed: {e}"}

    def _execute_stop(self) -> Dict[str, Any]:
        """Stop all robot motion."""
        try:
            self._wrapper.stop()
            return {"ok": True, "telemetry": {"stopped": True}}
        except Exception as e:
            return {"ok": False, "error": f"Stop failed: {e}"}

    def _execute_power_on(self) -> Dict[str, Any]:
        """Power on the robot."""
        try:
            if not self._wrapper.is_powered_on():
                self._wrapper.power_on()
                time.sleep(2)  # Wait for power up
            return {"ok": True, "telemetry": {"powered_on": True}}
        except Exception as e:
            return {"ok": False, "error": f"Power on failed: {e}"}

    def _execute_power_off(self) -> Dict[str, Any]:
        """Power off the robot."""
        try:
            self._wrapper.power_off()
            return {"ok": True, "telemetry": {"powered_off": True}}
        except Exception as e:
            return {"ok": False, "error": f"Power off failed: {e}"}

    def _execute_estop(self) -> Dict[str, Any]:
        """Engage emergency stop."""
        try:
            self._wrapper.assertEStop(True)
            self._estop_engaged = True
            return {"ok": True, "telemetry": {"estop": True}}
        except Exception as e:
            return {"ok": False, "error": f"E-Stop failed: {e}"}

    def _execute_estop_release(self) -> Dict[str, Any]:
        """Release emergency stop."""
        try:
            self._wrapper.assertEStop(False)
            self._estop_engaged = False
            return {"ok": True, "telemetry": {"estop_released": True}}
        except Exception as e:
            return {"ok": False, "error": f"E-Stop release failed: {e}"}

    def _execute_open_gripper(self) -> Dict[str, Any]:
        """Open the gripper."""
        try:
            self._wrapper.open_gripper()
            return {"ok": True, "telemetry": {"gripper": "open"}}
        except Exception as e:
            return {"ok": False, "error": f"Open gripper failed: {e}"}

    def _execute_close_gripper(self) -> Dict[str, Any]:
        """Close the gripper."""
        try:
            self._wrapper.close_gripper()
            return {"ok": True, "telemetry": {"gripper": "closed"}}
        except Exception as e:
            return {"ok": False, "error": f"Close gripper failed: {e}"}

    def _execute_arm_stow(self) -> Dict[str, Any]:
        """Stow the arm."""
        try:
            self._wrapper.stow_arm()
            return {"ok": True, "telemetry": {"arm": "stowed"}}
        except Exception as e:
            return {"ok": False, "error": f"Arm stow failed: {e}"}

    def _execute_arm_ready(self) -> Dict[str, Any]:
        """Move arm to ready position."""
        try:
            self._wrapper.unstow_arm()
            return {"ok": True, "telemetry": {"arm": "ready"}}
        except Exception as e:
            return {"ok": False, "error": f"Arm ready failed: {e}"}

    def _execute_take_picture(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Take picture from specified cameras."""
        try:
            # Default to front cameras if none specified
            cameras = params.get("cameras", ["frontleft_fisheye_image", "frontright_fisheye_image"])
            
            images = {}
            for camera in cameras:
                image = self._wrapper.get_image_from_sources([camera])
                if image:
                    images[camera] = f"captured_{int(time.time())}"
            
            return {"ok": True, "telemetry": {"images": images}}
        except Exception as e:
            return {"ok": False, "error": f"Take picture failed: {e}"}

    def _execute_dock(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Dock at charging station."""
        try:
            dock_id = params.get("dock_id", 520)  # Default dock ID
            success = self._wrapper.dock(dock_id)
            
            if success:
                return {"ok": True, "telemetry": {"docked": True, "dock_id": dock_id}}
            else:
                return {"ok": False, "error": f"Docking to {dock_id} failed"}
        except Exception as e:
            return {"ok": False, "error": f"Dock command failed: {e}"}

    def status(self) -> Dict[str, Any]:
        """Get comprehensive robot status."""
        self.ensure_connected()
        
        try:
            # Get robot state from wrapper
            robot_state = self._wrapper.get_robot_state()
            
            status = {
                "ok": True,
                "connected": self._wrapper is not None,
                "busy": self._busy,
                "powered_on": self._wrapper.is_powered_on() if self._wrapper else False,
                "standing": self._wrapper.is_standing() if self._wrapper else False,
                "estop_state": "engaged" if self._estop_engaged else "disengaged",
            }
            
            # Add battery information if available
            if robot_state and hasattr(robot_state, 'power_state'):
                if hasattr(robot_state.power_state, 'locomotion_charge_percentage'):
                    status["battery_pct"] = robot_state.power_state.locomotion_charge_percentage
            
            # Add gripper state if available
            try:
                gripper_state = self._wrapper.get_gripper_open_percentage()
                status["gripper_open_pct"] = gripper_state
            except:
                pass
            
            # Add location if available
            try:
                if hasattr(self._wrapper, 'get_xy_yaw'):
                    x, y, yaw = self._wrapper.get_xy_yaw()
                    status["pose"] = {"x": x, "y": y, "yaw": yaw}
            except:
                pass
            
            return status
            
        except Exception as e:
            return {
                "ok": False,
                "connected": False,
                "error": f"Status check failed: {e}",
                "busy": self._busy
            }

    def close(self) -> None:
        """Clean shutdown of Spot connection."""
        if self._wrapper is not None:
            try:
                # Sit the robot before disconnecting
                if self._wrapper.is_standing():
                    self._wrapper.sit()
                    time.sleep(1)
                
                # Power off if requested
                # self._wrapper.power_off()  # Uncomment if desired
                
                # Release E-Stop
                if self._estop_engaged:
                    self._wrapper.assertEStop(False)
                
                # Shutdown wrapper
                self._wrapper.shutdown()
            except Exception as e:
                print(f"Warning: Error during shutdown: {e}")
            finally:
                self._wrapper = None