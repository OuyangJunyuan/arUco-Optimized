############################################################################################
##                                          常用宏
##                                      ${PROJECT_NAME} 
##										${PROJECT_SOURCE_DIR} 本文件地址
##										${PROJECT_BINARY_DIR})  编译地址
##											输出信息
##										message(STATUS "xxx ${xxx}")
############################################################################################
cmake_minimum_required(VERSION 3.0.2)
##需和xml文件中的name一致
project(aruco_test)
#set( CMAKE_BUILD_TYPE "Release" )  #release模式
#set( CMAKE_CXX_FLAGS "-std=c++11 -O3" ) #c++11   o3优化


############################################################################################
##                                      non-ros packages:具体路径宏可以查询<库名Config.cmake>查看定义
## <name>_FOUND
## <name>_INCLUDE_DIRS  /  <name>_DIRS
## <name>_LIBRARIES  /  <name>_LIBS  /  <name>_LIBRARY
############################################################################################
##                          当前目录下放的头文件
#####################################################
#添加头文件目录 
include_directories(''./include'')  #相对于CMakeLists.txt
#添加库文件目录  
link_directories(" ./lib")  #相对于执行CMake的目录或路径一般是在
link_directories("/usr/lib/x86_64-linux-gnu/")
#添加所有源文件名到SRC_DIRS中
aux_source_directory(./src  SRCS_FILES)
#EXECUTABLE_OUTPUT_PATH(./bin)
#LIBRARY_OUTPUT_PATH(./lib)



##                          Eigen
#find_package(Eigen3  REQUIRED)
#include_directories(${Eigen3_INCLUDE_DIRS})
#list(APPEND ALL_LIBS ${Eigen3_LIBRARIES})

##                          Boost
#find_package(Boost REQUIRED COMPONENTS
# system  # format 
#)
#include_directories(${Boost_INCLUDE_DIRS})
#list(APPEND ALL_LIBS ${Boost_LIBRARIES})

##                          OpenCV
find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
list(APPEND ALL_LIBS ${OpenCV_LIBRARIES})

##                          Sophus
#find_package(Sophus REQUIRED) 
#include_directories(${Sophus_INCLUDE_DIRS}) #link  ${Sophus_LIBRARIES}
#list(APPEND ALL_LIBS ${Sophus_LIBRARIES})

##							Ceres
#find_package( Ceres REQUIRED )
#include_directories( ${CERES_INCLUDE_DIRS} ) #link  ${CERES_LIBRARIES}
#list(APPEND ALL_LIBS ${CERES_LIBRARIES})

##                         G2O
#find_package(G2O REQUIRED)
#include_directories(${G2O_INCLUDE_DIRS}) 
#list(APPEND ALL_LIBS ${G2O_LIBRARIES})

##                         glog
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /usr/local/lib/cmake)
#find_package(Glog REQUIRED)
#include_directories(${GLOG_INCLUDE_DIRS}) #link   ${GLOG_LIBRARIES}
#list(APPEND ALL_LIBS ${GLOG_LIBRARIES})

##                         gtest
#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIRS}) #link  ${GTEST_BOTH_LIBRARIES}
#list(APPEND ALL_LIBS ${GTEST_LIBRARIES})

##                         gflags
#find_package(GFlags REQUIRED)
#include_directories(${GFLAGS_INCLUDE_DIRS}) #link  ${GFLAGS_LIBRARIES}
#list(APPEND ALL_LIBS ${GFLAGS_LIBRARIES})

############################################################################################
##                                       build
############################################################################################
##                     generate libs like .s .so
#####################################################
MESSAGE(STATUS "This is BINARY dir " ${HELLO_BINARY_DIR})
MESSAGE(STATUS "This is SOURCE dir " ${HELLO_SOURCE_DIR})
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/mypackage.cpp
# )
##库内有用到自定义消息类型时启用,否则无法生成消息头文件
#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

#####################################################
##                          node
#####################################################
#add_executable(${PROJECT_NAME}_node src/main.cpp)
###可执行文件内有用到自定义消息类型时启用,否则无法生成消息头文件
##add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
#target_link_libraries(${PROJECT_NAME}_node
#        ${catkin_LIBRARIES}
#        )

#####################################################
##                         non-node
#####################################################
add_executable(${PROJECT_NAME}_exe ${SRCS_FILES})
target_link_libraries(${PROJECT_NAME}_exe
        ${OpenCV_LIBRARIES}  
        )
