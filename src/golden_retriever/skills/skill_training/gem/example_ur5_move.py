from retriever.robots.ur5_hhl.env import Env
from retriever.robots.ur5_hhl.ur5 import UR5


def setup_ros_dependencies():
    global rospy
    import rospy


if __name__ == "__main__":
    setup_ros_dependencies()

    rospy.init_node("example_move_to_position")

    # Config
    # Note: number of 'r' stands for number of rotation axes
    action_sequence = "pxyzrrr"
    obs_source = "raw"
    block_clip_processor = False
    kernel_size = 60
    stride = 20

    # Constants
    ws_center = [-0.445, -0.079, -0.1625]

    use_ur5_class = True
    use_env_class = False

    # NOTE: use UR5 class
    if use_ur5_class:
        pick_offset = 0.02
        place_offset = 0.1
        place_open_pos = 0.0

        ur5 = UR5(pick_offset, place_offset, place_open_pos)

        # Move to home
        print("Move to home pose")
        rospy.sleep(0.5)
        ur5.moveToHome()
        print("Done")

        # Move to a certain SE(3) pose
        print("Move to a SE(3) pose")
        rospy.sleep(0.5)
        ur5.moveToP(x=-0.445, y=-0.079, z=0.05, rx=0.0, ry=0.0, rz=0.0)
        print("Done")

    # NOTE: use Env instance with UR5 instance, need all camera setup
    if use_env_class:
        env = Env(
            ws_center=ws_center,
            action_sequence=action_sequence,
            obs_source=obs_source,
            block_clip_processor=block_clip_processor,
            kernel_size=kernel_size,
            stride=stride,
        )

        pixel_small_reso, pixel_big_reso = (
            env.cloud_proxy.img_width,
            env.cloud_proxy.img_height,
        )

        # Move to home
        print("Move to home pose")
        rospy.sleep(0.5)
        env.ur5.moveToHome()
        print("Done")

        # Move to a certain SE(3) pose
        print("Move to a SE(3) pose")
        rospy.sleep(0.5)
        # env.ur5.moveToP(x=-0.445, y=-0.079, z=-0.1625, rx=0.0, ry=0.0, rz=np.pi, v=0.1)
        # env.ur5.moveToP(x=-0.445, y=-0.079, z=-0.0625, rx=0.0, ry=0.0, rz=np.pi, v=0.1)
        # env.ur5.moveToP(x=-0.445, y=-0.079, z=0.05, rx=0.0, ry=0.0, rz=np.pi)
        env.ur5.moveToP(x=-0.445, y=-0.079, z=0.05, rx=0.0, ry=0.0, rz=0.0)
        print("Done")
