from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
import time

try:  # Optional dependency; runtime-checked
    import bosdyn.client  # type: ignore
    from bosdyn.client import create_standard_sdk  # type: ignore
    from bosdyn.client.util import authenticate  # type: ignore
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive  # type: ignore
    from bosdyn.client.robot_command import (  # type: ignore
        RobotCommandClient,
        RobotCommandBuilder,
        blocking_stand,
    )
    from bosdyn.client.power import PowerClient  # type: ignore
    from bosdyn.client.robot_state import RobotStateClient  # type: ignore
    from bosdyn.client.estop import EstopClient, EstopEndpoint  # type: ignore
    BOSDYN_AVAILABLE = True
except Exception:  # pragma: no cover - library may be absent in dev
    BOSDYN_AVAILABLE = False


@dataclass
class SpotConnectionConfig:
    host: str
    username: Optional[str] = None
    password: Optional[str] = None
    app_token: Optional[str] = None
    client_name: str = "retriever-spot"
    heartbeat_sec: float = 1.0
    command_timeout_sec: float = 10.0


class SpotConnectionError(RuntimeError):
    pass


from retriever.robots.connection_base import RobotConnection


class SpotConnectionManager(RobotConnection):
    """Owns the Spot SDK client and maintains connection lifecycle.

    Non-serializable SDK objects are confined here; callers interact via simple methods.
    """

    def __init__(self, cfg: SpotConnectionConfig):
        self.cfg = cfg
        self._sdk = None
        self._robot = None
        self._last_heartbeat = 0.0
        # Spot clients (initialized post-auth)
        self._lease_client = None
        self._lease_keepalive = None
        self._cmd_client = None
        self._state_client = None
        self._power_client = None
        self._estop_client = None
        self._estop_endpoint = None
        # Simple in-process queue controls (backpressure/cancel stubs)
        self._busy: bool = False
        self._cancel_flag: bool = False
        self._estop_engaged: bool = False  # mock/status hint

    def connect(self) -> None:
        if not BOSDYN_AVAILABLE:
            raise SpotConnectionError(
                "Boston Dynamics Spot SDK not installed. Install spot deps or use MockSpotConnectionManager."
            )
        if self._robot is not None:
            return
        self._sdk = create_standard_sdk(self.cfg.client_name)
        self._robot = self._sdk.create_robot(self.cfg.host)
        if self.cfg.app_token:
            # Token authentication path
            self._robot.authenticate_with_token(self.cfg.app_token)
        else:
            if not (self.cfg.username and self.cfg.password):
                raise SpotConnectionError("Username/password required when no app_token is provided.")
            authenticate(self._robot, self.cfg.username, self.cfg.password)
        # Initialize clients and lease
        self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)
        # KeepAlive maintains the lease in the background
        self._lease_keepalive = LeaseKeepAlive(self._lease_client, return_at_exit=True)
        self._cmd_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
        self._state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
        try:
            self._power_client = self._robot.ensure_client(PowerClient.default_service_name)
        except Exception:
            self._power_client = None
        try:
            self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
        except Exception:
            self._estop_client = None
        # Ensure time sync and basic posture
        self._robot.time_sync.wait_for_sync()
        try:
            blocking_stand(self._cmd_client, timeout_sec=10)
        except Exception:
            # Standing may fail if already standing or arm-less; non-fatal
            pass
        self._last_heartbeat = time.time()

    def ensure_connected(self) -> None:
        if self._robot is None:
            self.connect()
        # Cheap heartbeat throttle
        now = time.time()
        if now - self._last_heartbeat >= self.cfg.heartbeat_sec:
            try:
                assert self._robot is not None
                # Heartbeat: verify time sync and that clients are usable
                self._robot.time_sync.wait_for_sync()
                if self._state_client is not None:
                    # Quick state fetch to validate connectivity; keep cheap
                    _ = self._state_client.get_robot_state()
                self._last_heartbeat = now
            except Exception as e:
                # Attempt reconnect on failure
                self._robot = None
                self._sdk = None
                self._lease_client = None
                self._lease_keepalive = None
                self._cmd_client = None
                self._state_client = None
                # Simple exponential backoff with cap
                backoff = getattr(self, "_backoff", self.cfg.heartbeat_sec)
                time.sleep(min(backoff, 5.0))
                self._backoff = min(backoff * 2.0, 5.0)
                self.connect()
                self._backoff = self.cfg.heartbeat_sec

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a high-level command dict.

        Expected schema: {"type": str, "parameters": dict}
        Returns: result dict with ok/telemetry fields.
        """
        self.ensure_connected()
        typ = command.get("type")
        params = command.get("parameters", {})
        # NOTE: Minimal stub to illustrate mapping. Real implementation should
        # call Spot services (robot_command, lease, etc.).
        try:
            # Backpressure: reject if already executing a long operation
            if self._busy:
                return {"ok": False, "error": "Busy", "retry_after_sec": self.cfg.heartbeat_sec}
            self._busy = True
            self._cancel_flag = False
            if typ in ("move_to", "move_velocity"):
                # Map to SE2 body frame move (relative)
                if BOSDYN_AVAILABLE and self._cmd_client is not None:
                    x = float(params.get("x", 0.0))
                    y = float(params.get("y", 0.0))
                    yaw = float(params.get("yaw", 0.0))
                    cmd = RobotCommandBuilder.synchro_se2_velocity_command(
                        v_x=x, v_y=y, v_rot=yaw
                    )
                    _ = self._cmd_client.robot_command(cmd, end_time_secs=time.time() + self.cfg.command_timeout_sec)
                    return {"ok": True, "telemetry": {"cmd": "se2_velocity", "x": x, "y": y, "yaw": yaw}}
                return {"ok": True, "telemetry": {"arrived": True}}  # fallback
            if typ in ("stand",):
                if BOSDYN_AVAILABLE and self._cmd_client is not None:
                    blocking_stand(self._cmd_client, timeout_sec=self.cfg.command_timeout_sec)
                    return {"ok": True, "telemetry": {"standing": True}}
                return {"ok": True, "telemetry": {"standing": True}}
            if typ in ("move_trajectory",):
                # Draft: map to SE2 trajectory command when SDK is available
                if BOSDYN_AVAILABLE and self._cmd_client is not None:
                    try:
                        # Preferred: sequence of trajectory points if waypoints provided
                        waypoints = params.get("waypoints")
                        frame = params.get("frame", "body")
                        if waypoints and isinstance(waypoints, list):
                            from bosdyn.client import math_helpers  # type: ignore
                            for wp in waypoints:
                                x = float(wp.get("x", 0.0))
                                y = float(wp.get("y", 0.0))
                                yaw = float(wp.get("yaw", 0.0))
                                duration = float(wp.get("duration_sec", 1.0))
                                cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                                    goal_x=x,
                                    goal_y=y,
                                    goal_heading=yaw,
                                    frame_name=frame,
                                    time_since_reference=duration,
                                )
                                _ = self._cmd_client.robot_command(
                                    cmd, end_time_secs=time.time() + self.cfg.command_timeout_sec
                                )
                            return {"ok": True, "telemetry": {"cmd": "se2_trajectory", "waypoints": len(waypoints), "frame": frame}}
                        else:
                            # Single point or velocity fallback
                            x = float(params.get("x", 0.0))
                            y = float(params.get("y", 0.0))
                            yaw = float(params.get("yaw", 0.0))
                            duration = float(params.get("duration_sec", 2.0))
                            try:
                                cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                                    goal_x=x,
                                    goal_y=y,
                                    goal_heading=yaw,
                                    frame_name=frame,
                                    time_since_reference=duration,
                                )
                            except Exception:
                                cmd = RobotCommandBuilder.synchro_se2_velocity_command(v_x=x, v_y=y, v_rot=yaw)
                            _ = self._cmd_client.robot_command(
                                cmd, end_time_secs=time.time() + self.cfg.command_timeout_sec
                            )
                            return {"ok": True, "telemetry": {"cmd": "se2_trajectory", "x": x, "y": y, "yaw": yaw, "frame": frame}}
                    except Exception as e:
                        return {"ok": False, "error": f"trajectory failed: {e}"}
                return {"ok": True, "telemetry": {"cmd": "se2_trajectory_mock"}}
            if typ in ("sit",):
                if BOSDYN_AVAILABLE and self._cmd_client is not None:
                    try:
                        cmd = RobotCommandBuilder.synchro_sit_command()
                        _ = self._cmd_client.robot_command(cmd)
                        return {"ok": True, "telemetry": {"sitting": True}}
                    except Exception as e:
                        return {"ok": False, "error": f"sit failed: {e}"}
                return {"ok": False, "error": "sit unsupported without SDK"}
            if typ in ("power_on",):
                if BOSDYN_AVAILABLE and self._robot is not None:
                    self._robot.power_on(timeout_sec=self.cfg.command_timeout_sec)
                    return {"ok": True, "telemetry": {"powered_on": True}}
                return {"ok": True, "telemetry": {"powered_on": True}}
            if typ in ("power_off",):
                if BOSDYN_AVAILABLE and self._robot is not None:
                    self._robot.power_off(cut_immediately=False)
                    return {"ok": True, "telemetry": {"powered_off": True}}
                return {"ok": True, "telemetry": {"powered_off": True}}
            if typ == "open_gripper":
                # Placeholder: proper mapping would use manipulation API
                return {"ok": True, "telemetry": {"gripper": "open"}}
            if typ == "close_gripper":
                return {"ok": True, "telemetry": {"gripper": "closed"}}
            if typ in ("arm_stow",):
                # Placeholder mapping for arm stow
                return {"ok": True, "telemetry": {"arm": "stowed"}}
            if typ in ("arm_ready",):
                # Placeholder mapping for arm ready position
                return {"ok": True, "telemetry": {"arm": "ready"}}
            if typ in ("arm_move_cartesian",):
                # Placeholder: move arm cartesian to pose; expects x,y,z,roll,pitch,yaw
                pose = {k: float(params.get(k, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]}
                return {"ok": True, "telemetry": {"arm_pose": pose}}
            if typ == "estop" or typ == "e_stop":
                # Draft: prefer power off if estop client not configured.
                if BOSDYN_AVAILABLE and self._estop_client is not None:
                    try:
                        if self._estop_endpoint is None:
                            self._estop_endpoint = EstopEndpoint(self._estop_client, "retriever-estop", timeout=1.0)
                        self._estop_endpoint.force_simple_setup()
                        self._estop_endpoint.estop()
                        self._estop_engaged = True
                        return {"ok": True, "telemetry": {"estop": True}}
                    except Exception as e:
                        # Fallback to power off
                        try:
                            if self._robot is not None:
                                self._robot.power_off(cut_immediately=True)
                            self._estop_engaged = True
                            return {"ok": True, "telemetry": {"powered_off": True, "via": "fallback"}}
                        except Exception:
                            return {"ok": False, "error": f"estop failed: {e}"}
                else:
                    # Fallback to immediate power off
                    if self._robot is not None:
                        self._robot.power_off(cut_immediately=True)
                        self._estop_engaged = True
                        return {"ok": True, "telemetry": {"powered_off": True}}
                    return {"ok": False, "error": "no robot handle for power off"}
            if typ == "stop":
                if BOSDYN_AVAILABLE and self._cmd_client is not None:
                    cmd = RobotCommandBuilder.stop_command()
                    _ = self._cmd_client.robot_command(cmd)
                    return {"ok": True, "telemetry": {"stopped": True}}
                return {"ok": True, "telemetry": {"stopped": True}}
            if typ in ("estop_release", "estop_disengage"):
                # Draft: attempt to clear local flag and power on
                self._estop_engaged = False
                if BOSDYN_AVAILABLE and self._robot is not None:
                    try:
                        self._robot.power_on(timeout_sec=self.cfg.command_timeout_sec)
                        return {"ok": True, "telemetry": {"estop_released": True, "powered_on": True}}
                    except Exception as e:
                        return {"ok": False, "error": f"release failed: {e}"}
                return {"ok": True, "telemetry": {"estop_released": True}}
            return {"ok": False, "error": f"Unknown command type: {typ}"}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "error": str(e)}
        finally:
            self._busy = False

    def status(self) -> Dict[str, Any]:
        self.ensure_connected()
        status: Dict[str, Any] = {"ok": True, "connected": self._robot is not None}
        status["busy"] = self._busy
        if BOSDYN_AVAILABLE and self._state_client is not None:
            try:
                rs = self._state_client.get_robot_state()
                # Battery (first battery as a simplification)
                if rs.power_state.locomotion_charge_percentage is not None:
                    status["battery_pct"] = rs.power_state.locomotion_charge_percentage
                # EStop state (simplified)
                status["estop_state"] = str(getattr(rs.estop_state, "status", "unknown"))
                # Faults summary
                faults = list(getattr(rs, "faults", []))
                status["faults_count"] = len(faults)
                # Include a brief summary up to 5 entries
                try:
                    status["faults"] = [getattr(f, "onset_fault", None) for f in faults[:5]]
                except Exception:
                    status["faults"] = []
                # Power state
                status["powered_on"] = bool(getattr(rs.power_state, "motor_power_state", 0))
            except Exception:
                pass
        else:
            # Mock hint
            status["estop_state"] = "engaged" if self._estop_engaged else "disengaged"
        return status

    def close(self) -> None:
        # Spot SDK typically does not require explicit close; add hooks for leases/session if needed.
        if self._lease_keepalive is not None:
            try:
                self._lease_keepalive.shutdown()
            except Exception:
                pass
        self._lease_keepalive = None
        self._lease_client = None
        self._cmd_client = None
        self._state_client = None
        self._power_client = None
        self._estop_client = None
        self._estop_endpoint = None
        self._robot = None
        self._sdk = None

    # Optional: cooperative cancel for long-running ops
    def cancel_current(self) -> None:
        self._cancel_flag = True


class MockSpotConnectionManager(SpotConnectionManager):
    """In-memory mock for tests and local development without hardware."""

    def __init__(self, cfg: SpotConnectionConfig):
        super().__init__(cfg)
        self._robot = object()  # sentinel
        self._sdk = object()
        self._pose = (0.0, 0.0, 0.0)
        self._gripper = "open"

    def connect(self) -> None:
        # Always "connected"
        self._last_heartbeat = time.time()

    def execute(self, command: Dict[str, Any]) -> Dict[str, Any]:
        typ = command.get("type")
        params = command.get("parameters", {})
        if typ == "move_to":
            x = float(params.get("x", 0.0))
            y = float(params.get("y", 0.0))
            yaw = float(params.get("yaw", 0.0))
            self._pose = (x, y, yaw)
            return {"ok": True, "telemetry": {"pose": self._pose}}
        if typ == "open_gripper":
            self._gripper = "open"
            return {"ok": True, "telemetry": {"gripper": self._gripper}}
        if typ == "close_gripper":
            self._gripper = "closed"
            return {"ok": True, "telemetry": {"gripper": self._gripper}}
        if typ == "stop":
            return {"ok": True, "telemetry": {"stopped": True}}
        if typ == "stand":
            return {"ok": True, "telemetry": {"standing": True}}
        if typ == "sit":
            return {"ok": True, "telemetry": {"sitting": True}}
        if typ == "move_trajectory":
            return {"ok": True, "telemetry": {"cmd": "se2_trajectory_mock"}}
        if typ == "power_on":
            return {"ok": True, "telemetry": {"powered_on": True}}
        if typ == "power_off":
            return {"ok": True, "telemetry": {"powered_off": True}}
        if typ in ("estop", "e_stop"):
            self._estop_engaged = True
            return {"ok": True, "telemetry": {"estop": True}}
        if typ in ("estop_release", "estop_disengage"):
            self._estop_engaged = False
            return {"ok": True, "telemetry": {"estop_released": True}}
        if typ in ("arm_stow",):
            return {"ok": True, "telemetry": {"arm": "stowed"}}
        if typ in ("arm_ready",):
            return {"ok": True, "telemetry": {"arm": "ready"}}
        return {"ok": False, "error": f"Unknown command type: {typ}"}

    def status(self) -> Dict[str, Any]:
        """Return a lightweight status snapshot compatible with safety flows.

        Include an `estop_state` field so EstopStatusMonitorFlow can work with
        this mock just like with a real manager implementation.
        """
        return {
            "ok": True,
            "connected": True,
            "pose": self._pose,
            "gripper": self._gripper,
            "estop_state": "engaged" if getattr(self, "_estop_engaged", False) else "disengaged",
        }
