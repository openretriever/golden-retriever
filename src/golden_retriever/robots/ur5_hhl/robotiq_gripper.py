# Adopted from code by Dian, Xupeng, Haojie, Mingxi

"""
This module represents a robot gripper. It uses the gripper driver from
https://github.com/UTNuclearRoboticsPublic/robotiq/tree/kinetic-devel.
"""
import numpy as np


def setup_ros_dependencies():
    global rospy, GripperCmd, GripperStat
    import rospy
    from robotiq_c_model_control.msg import CModel_robot_input as GripperStat
    from robotiq_c_model_control.msg import CModel_robot_output as GripperCmd


class Gripper:
    def __init__(self, isMoving):
        """Constructor."""
        setup_ros_dependencies()

        self.status = None
        self.isMoving = isMoving
        self.speed = 10
        self.speed_window_size = 2
        self.speed_window = np.zeros([self.speed_window_size])  # [new, old, older, ...]

        self.gripperSub = rospy.Subscriber(
            "/CModelRobotInput", GripperStat, self.updateGripperStat
        )
        self.gripperPub = rospy.Publisher(
            "/CModelRobotOutput", GripperCmd, queue_size=1
        )

        print("Waiting for gripper driver to connect ...")
        while self.gripperPub.get_num_connections() == 0 or self.status is None:
            rospy.sleep(0.01)

        if self.status.gACT == 0:
            self.reset()
            self.activate()

    def updateGripperStat(self, msg):
        """Obtain the status of the gripper."""

        self.status = msg
        self.speed_window[1:] = self.speed_window[:-1]
        self.speed_window[0] = self.status.gPO
        self.speed = (
            self.speed_window[-1] - self.speed_window[0]
        ) / self.speed_window_size
        # print(self.speed_window[-1] - self.speed_window[0])
        # print(self.speed_window)

    def reset(self):
        """Reset the gripper."""

        print("Resetting gripper ...")

        cmd = GripperCmd()
        cmd.rACT = 0
        self.gripperPub.publish(cmd)
        rospy.sleep(0.5)

    def activate(self):
        """Activate the gripper."""

        print("Activating gripper ...")

        cmd = GripperCmd()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rSP = 255
        cmd.rFR = 150
        self.gripperPub.publish(cmd)
        rospy.sleep(0.5)

    def closeGripper(self, speed=255, force=255):
        """Close the gripper. Default values for optional arguments are set to their max."""

        if not self.isMoving:
            return

        # print('Closing gripper ...')
        cmd = GripperCmd()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = 255  # position
        cmd.rFR = force
        cmd.rSP = speed
        self.gripperPub.publish(cmd)
        # rospy.sleep(0.5)

    def openGripper(self, speed=255, force=255, position=0):
        """Open the gripper. Default values for optional arguments are set to their max."""

        if not self.isMoving:
            return

        # print('Opening gripper ...')
        cmd = GripperCmd()
        cmd.rACT = 1
        cmd.rGTO = 1
        cmd.rPR = position  # position
        cmd.rFR = force
        cmd.rSP = speed
        self.gripperPub.publish(cmd)
        # rospy.sleep(0.5)

    def hasObj(self, wait_speed0=False):
        # return self.status.gCU < 10
        # return self.status.gPO > 204
        # rospy.sleep(0.4)
        while True:
            # print(self.speed, self.status.gOBJ, self.speed_window)
            if self.status.gOBJ != 0 and (not wait_speed0 or self.speed == 0):
                break
            else:
                rospy.sleep(0.05)

        return self.status.gOBJ == 2  # 0 is moving, 3 fully opened or closed
