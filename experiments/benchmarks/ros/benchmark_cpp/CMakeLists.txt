cmake_minimum_required(VERSION 3.20)
project(benchmark_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)

# Shared component library — both nodes registered for composition / intra-process.
add_library(${PROJECT_NAME}_components SHARED
  src/publisher.cpp
  src/subscriber.cpp)
target_compile_definitions(${PROJECT_NAME}_components PRIVATE BENCHMARK_CPP_AS_COMPONENT)
target_link_libraries(${PROJECT_NAME}_components PUBLIC
  rclcpp::rclcpp
  rclcpp_components::component
  rclcpp_components::component_manager
  ${std_msgs_TARGETS})
rclcpp_components_register_nodes(${PROJECT_NAME}_components
  "MinimalPublisher"
  "MinimalSubscriber")

# Standalone executables (same translation units; main() is inert when loaded as a component).
add_executable(publisher src/publisher.cpp)
target_link_libraries(publisher PUBLIC 
  rclcpp::rclcpp
  rclcpp_components::component
  rclcpp_components::component_manager
  ${std_msgs_TARGETS}
)

add_executable(subscriber src/subscriber.cpp)
target_link_libraries(subscriber PUBLIC
  rclcpp::rclcpp
  rclcpp_components::component
  rclcpp_components::component_manager
  ${std_msgs_TARGETS}
)

install(TARGETS ${PROJECT_NAME}_components
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(TARGETS
  publisher
  subscriber
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
