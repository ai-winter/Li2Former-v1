cmake_minimum_required(VERSION 2.8.3)
project(ped_real)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
)

catkin_package(
    
)

catkin_install_python(PROGRAMS
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)