#!/usr/bin/env python
"""Broadcasts the transform for the sensor.

Credits:
- http://wiki.ros.org/tf/Tutorials/Adding%20a%20frame%20%28Python%29

Assembled: Northeastern University, 2015
"""


def setup_ros_dependencies():
    global rospy, tf
    import rospy
    import tf


if __name__ == "__main__":
    setup_ros_dependencies()

    rospy.init_node("add_sensor_frame")
    br = tf.TransformBroadcaster()
    rate = rospy.Rate(50.0)

    while not rospy.is_shutdown():
        # measured
        # br.sendTransform((0.092, 0.062, 0.044), (0, 0, 0, 1), rospy.Time.now(), "camera_link", "ee_link")
        # block calibrated

        # br.sendTransform(
        #     (0.010, 0.095, 1.047),
        #     (0.002, 1.000, -0.011, 0.024),
        #     rospy.Time.now(),
        #     "my_bundle_2",
        #     "cam_2_color_optical_frame",
        # )
        #
        # br.sendTransform(
        #     (0.166, -0.087, 1.250),
        #     (0.957, -0.035, 0.288, -0.002),
        #     rospy.Time.now(),
        #     "my_bundle_4",
        #     "cam_4_color_optical_frame",
        # )
        # br.sendTransform(
        #     (-0.017, -0.117, 1.045),
        #     (1.000, 0.002, -0.018, -0.003),
        #     rospy.Time.now(),
        #     "my_bundle_5",
        #     "cam_5_color_optical_frame",
        # )
        # br.sendTransform(
        #     (-0.158, -0.050, 1.257),
        #     (0.965, 0.017, -0.260, 0.003),
        #     rospy.Time.now(),
        #     "my_bundle_6",
        #     "cam_6_color_optical_frame",
        # )
        #
        # br.sendTransform(
        #     (-0.819, -0.033, 0.954),
        #     (0.386, 0.262, 0.666, -0.582),
        #     rospy.Time.now(),
        #     "cam_4_link",
        #     "virtual_baselink",
        # )
        # br.sendTransform(
        #     (0.062, -0.123, 1.043),
        #     (0.509, 0.511, 0.490, -0.489),
        #     rospy.Time.now(),
        #     "cam_5_link",
        #     "virtual_baselink",
        # )
        # br.sendTransform(
        #     (0.889, 0.206, 0.158),
        #     (0.639, 0.661, -0.229, -0.321),
        #     rospy.Time.now(),
        #     "cam_6_link",
        #     "virtual_baselink",
        # )

        br.sendTransform(
            (-0.799, -0.044, 0.541),
            (-0.604, 0.351, 0.189, 0.691),
            rospy.Time.now(),
            "cam_4_link",
            "virtual_baselink",
        )
        br.sendTransform(
            (0.055, -0.119, 1.044),
            (-0.494, 0.506, 0.506, 0.493),
            rospy.Time.now(),
            "cam_5_link",
            "virtual_baselink",
        )
        br.sendTransform(
            (0.781, -0.044, 0.496),
            (-0.321, 0.639, 0.661, 0.228),
            rospy.Time.now(),
            "cam_6_link",
            "virtual_baselink",
        )

        rate.sleep()
