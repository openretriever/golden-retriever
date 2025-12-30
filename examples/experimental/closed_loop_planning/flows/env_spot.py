import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from retriever.flow import Flow, flow_io

from ..types.flow_types import EnvInput, EnvOutput

logger = logging.getLogger(__name__)

# Check for Bosdyn SDK
try:
    import bosdyn.api.image_pb2 as image_pb2
    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client.estop import EstopClient
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.robot_command import RobotCommandClient
    from bosdyn.client.robot_state import RobotStateClient
    from bosdyn.client.image import ImageClient 
    from bosdyn.api import image_pb2
    BOSDYN_AVAILABLE = True
except ImportError:
    BOSDYN_AVAILABLE = False
    logger.warning("Bosdyn SDK not found. SpotEnvironmentFlow will be non-functional.")

@flow_io
@dataclass
class SpotEnvOutput(EnvOutput):
    """Output from the Real Spot environment."""
    data: dict = None
    # Potentially add specific bosdyn state objects here if needed,
    # but untyped dict is safer for serialization unless we wrap them.

class SpotEnvironmentFlow(Flow[EnvInput, SpotEnvOutput]):
    """Real Spot Environment Flow.
    
    Connects to a real Boston Dynamics Spot robot using the Bosdyn SDK.
    Requires BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD, and SPOT_IP env vars.
    """

    def __init__(self, name: str = "SpotEnvironmentFlow", hostname: Optional[str] = None):
        self.name = name
        self.hostname = hostname or os.environ.get("SPOT_IP")
        self.username = os.environ.get("BOSDYN_CLIENT_USERNAME", "user")
        self.password = os.environ.get("BOSDYN_CLIENT_PASSWORD", "password")

        self.sdk = None
        self.robot = None
        self.command_client = None
        self.robot_state_client = None
        self.lease_client = None
        self.lease_keepalive = None

        if BOSDYN_AVAILABLE and self.hostname:
            self._initialize_robot()
        else:
            logger.warning(f"[{self.name}] Skipped initialization (SDK={BOSDYN_AVAILABLE}, Host={self.hostname})")

    def _initialize_robot(self):
        try:
            logger.info(f"[{self.name}] Connecting to Spot at {self.hostname}...")
            self.sdk = bosdyn.client.create_standard_sdk('SpotClient')
            self.robot = self.sdk.create_robot(self.hostname)
            self.robot.authenticate(self.username, self.password)

            # Start time sync
            self.robot.time_sync.wait_for_sync()

            # Create clients
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name) # Initialize image client here

            logger.info(f"[{self.name}] Connected to Spot.")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to connect: {e}")
            raise

    def reset(self):
        if not self.robot:
            logger.warning(f"[{self.name}] Reset called but robot not connected.")
            return

        # Take Lease
        self.lease_client.take()
        self.lease_keepalive = LeaseKeepAlive(self.lease_client)

        # Power On
        if not self.robot.is_powered_on():
            logger.info(f"[{self.name}] Powering on robot...")
            self.robot.power_on(timeout_sec=20)
            logger.info(f"[{self.name}] Robot powered on.")

        # Stand
        logger.info(f"[{self.name}] Commanding stand...")
        from bosdyn.client.robot_command import RobotCommandBuilder
        cmd = RobotCommandBuilder.synchro_stand_command(params=None)
        self.command_client.robot_command(cmd)
        logger.info(f"[{self.name}] Robot standing.")

    def step(self, inp: EnvInput) -> SpotEnvOutput:
        if not self.robot:
            return SpotEnvOutput(data={"status": "not_connected"})

        # 1. Execute Action
        if inp.action:
            from bosdyn.client.robot_command import RobotCommandBuilder

            # Simple mapping: if action has array [dx, dy], treat as velocity command
            if inp.action.arr is not None and len(inp.action.arr) >= 2:
                v_x, v_y = inp.action.arr[0], inp.action.arr[1]
                yaw_rate = inp.action.arr[2] if len(inp.action.arr) > 2 else 0.0

                logger.info(f"[{self.name}] Velocity Cmd: vx={v_x:.2f}, vy={v_y:.2f}, w={yaw_rate:.2f}")

                cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=yaw_rate)
                self.command_client.robot_command(cmd, end_time_secs=time.time() + 0.5) # Execute for short duration

            else:
                logger.warning(f"[{self.name}] Action received but unhandled: {inp.action}")

        # 2. Get State
        robot_state = self.robot_state_client.get_robot_state()
        kinematic_state = robot_state.kinematic_state

        # Extract basic info
        pos = kinematic_state.transforms_snapshot.child_to_parent_edge_map["body"].parent_tform_child.position

        # Log to Rerun
        import rerun as rr
        # Log Robot position (assuming world frame is close to odom/vision frame for demo)
        rr.log("world/spot", rr.Points3D([[pos.x, pos.y, pos.z]], colors=[[255, 200, 0]], radii=[0.2], labels=["Spot"]))

        obs_data = {
            "robot_pos": [pos.x, pos.y, pos.z],
            "battery_states": [s.charge_percentage for s in robot_state.battery_states],
            "power_state": robot_state.power_state.motor_power_state,
            "image": image_data
        }

        return SpotEnvOutput(data=obs_data)

    def cleanup(self):
        if self.lease_keepalive:
            self.lease_keepalive.shutdown()
