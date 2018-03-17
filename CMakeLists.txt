cmake_minimum_required(VERSION 2.8.3)
project(top_mapping)

find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS main.py
                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})