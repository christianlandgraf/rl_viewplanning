cmake_minimum_required(VERSION 3.0.2)
project(ipa_kifz_viewplanning)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)
# set(CMAKE_CXX_FLAGS "-Wall -pedantic -Wextra -fopenmp")
# SET(qhullIncludeDir /usr/include/libqhull/)
# SET(qhullLibraryDir /usr/lib/x86_64-linux-gnu/)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED
  roscpp rospy std_msgs geometry_msgs genmsg
  sensor_msgs pcl_ros
  # tf2_eigen open3d_conversions
  # moveit_core moveit_ros_planning
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
# find_package(Eigen3 REQUIRED)
# find_package(Open3D REQUIRED)
find_package(PCL 1.10 REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

add_service_files(
   FILES
   GetAreaGain.srv
)

generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
  # LIBRARIES ipa_kifz_viewplanning
  # CATKIN_DEPENDS roscpp sensor_msgs open3d_conversions
  #  DEPENDS Open3D EIGEN3
  DEPENDS message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
#  ${Open3D_INCLUDE_DIRS}
#  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

link_directories(${PCL_LIBRARY_DIRS})

add_executable (point_cloud_handler include/point_cloud_handler.h src/point_cloud_handler.cpp)
target_link_libraries (point_cloud_handler ${PCL_LIBRARIES} ${catkin_LIBRARIES})


install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

