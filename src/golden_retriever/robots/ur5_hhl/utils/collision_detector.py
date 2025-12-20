# Adopted from code by Dian, Xupeng, Haojie, Mingxi


def setup_ros_dependencies():
    global rospy, WrenchStamped, String
    import rospy
    from geometry_msgs.msg import WrenchStamped
    from std_msgs.msg import String


class CollisionDetector:
    def __init__(self, max_z=20, stop_acc=20, n_readings=1):
        """
        max_z : maximum allowable force (N) in the positive z direction, it should
                be positive if you want to detect collisions when end effector
                is moving vertically down
        stop_acc : acceleration of stop command
        n_readings : number of consecutive force readings above max_z before the
                     motion is stopped, used to prevent single noisy measurements
                     causing premature stoppage
        """
        # Note: workaround for ROS dependencies
        setup_ros_dependencies()

        self.is_running = False
        self.alarm_level = 0
        self.max_alarm_level = n_readings

        self.stop_command = "stopl({})".format(stop_acc)
        self.command_pub = rospy.Publisher(
            "/ur_hardware_interface/script_command", String, queue_size=10
        )

        self.max_z = max_z
        self.wrench_sub = rospy.Subscriber(
            "/wrench", WrenchStamped, self.wrenchCallback
        )

    def __enter__(self):
        self.is_running = True
        self.alarm_level = 0
        return self

    def __exit__(self, *args):
        self.is_running = False

    def wrenchCallback(self, msg):
        """Callback function for the wrench ROS topic."""
        if not self.is_running:
            return
        # print(msg.wrench.force.z)

        # if msg.wrench.force.z > self.max_z:
        #     self.alarm_level += 1
        # else:
        #     self.alarm_level = 0

        # if self.alarm_level >= self.max_alarm_level:
        if msg.wrench.force.z > self.max_z:
            self.command_pub.publish(self.stop_command)
            # print(msg.wrench.force.z, ' > ', self.max_z)
            self.is_running = False

        # if msg.wrench.force.z > 5:
        #     print(msg.wrench.force.z)
