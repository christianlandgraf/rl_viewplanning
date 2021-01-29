cmake_minimum_required(VERSION 3.0.2)
project(ipa_kifz_description)

find_package(catkin REQUIRED)

catkin_package()

include_directories()
install(DIRECTORY urdf DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )
install(DIRECTORY meshes DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION} )