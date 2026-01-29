cmake_minimum_required(VERSION 3.20)
project(benchmark_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(publisher src/publisher.cpp)
target_link_libraries(publisher PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS})

add_executable(subscriber src/subscriber.cpp)
target_link_libraries(subscriber PUBLIC rclcpp::rclcpp ${std_msgs_TARGETS})

install(TARGETS
  publisher
  subscriber
  DESTINATION lib/${PROJECT_NAME})

ament_package()
