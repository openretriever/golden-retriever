
"""
This helper script is used to set up the UR5 robot on the control workstation.
"""

# (source file)
source devel/setup.bash

# robot
echo ">>>>> launch robot driver"
# TODO note: the address might change
# (roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.188.62.92 limited:=true headless_mode:=true) &
(roslaunch ur_robot_driver ur5_bringup.launch robot_ip:=10.188.62.130 limited:=true headless_mode:=true) &
sleep 5

# camera 1:
echo ">>>>> launch camera 1"
(roslaunch realsense2_camera rs_camera.launch camera:=cam_1 filters:=spatial,temporal,pointcloud serial_no:=215122256404 align_depth:=true) &
sleep 5

# camera 3:
echo ">>>>> launch camera 3"
(roslaunch realsense2_camera rs_camera.launch camera:=cam_3 filters:=spatial,temporal,pointcloud serial_no:=215122256436 align_depth:=true) &
sleep 5

# (gripper troubleshooting)
# sudo chmod 777 /dev/ttyUSB0
# start gripper:
echo ">>>>> launch gripper"
(rosrun robotiq_c_model_control CModelRtuNode.py /dev/ttyUSB0) &
sleep 5

# # camera 2 on another machine:
# # Note: this machine can connect to `ur5-earth` via SSH key directly without password
echo ">>>>> launch camera 2"
#(ssh ur5-earth 'source /opt/ros/melodic/setup.bash && source ~/rgbd_grasp_ws/devel/setup.bash && roslaunch azure_kinect_ros_driver kinect_rgbd.launch') &
#sleep 5
#(ssh ur5-earth 'source /opt/ros/melodic/setup.bash && source ~/rgbd_grasp_ws/devel/setup.bash && roslaunch azure_kinect_ros_driver driver.launch') &
sleep 5

# # Note: another solution, use screen to keep the process running after SSH session is closed
# # ssh ur5-earth 'screen -dmS cam2_launch_session bash -c "source /opt/ros/melodic/setup.bash && source ~/rgbd_grasp_ws/devel/setup.bash && roslaunch azure_kinect_ros_driver kinect_rgbd.launch"'
# sleep 5

# TF transform:
echo ">>>>> publish TF transform"
(python3 ~/rgbd_grasp_ws/src/helping_hands_rl_ur5/src/add_sensor_frame.py) &

wait