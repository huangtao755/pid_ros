# >>> PX4 initialize >>>
source ~/src/PX4-Autopilot/Tools/simulation/gazebo-classic/setup_gazebo.bash ~/src/PX4-Autopilot ~/src/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/src/PX4-Autopilot/Tools/simulation/gazebo-classic/sitl_gazebo-classic
# <<< PX4 initialize <<<


cd ~/src/PX4-Autopilot
DONT_RUN=1 make px4_sitl_default gazebo
roslaunch launch/mavros_posix_sitl.launch
