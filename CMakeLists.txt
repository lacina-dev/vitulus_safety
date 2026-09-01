cmake_minimum_required(VERSION 3.0.2)
project(vitulus_safety)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  vitulus_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2_ros
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs vitulus_msgs nav_msgs geometry_msgs visualization_msgs tf2_ros
)

catkin_install_python(PROGRAMS
  nodes/safety_node.py
  nodes/lidar_objects.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
