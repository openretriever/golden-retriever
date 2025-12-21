cmake_minimum_required(VERSION 3.16)
project(cpp_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# -----------------------------------------------------------------------------
# Build Dora C API from source (since it's not in pixi/conda yet)
# -----------------------------------------------------------------------------
message(STATUS "Fetching Dora 0.3.5 source for C bindings...")
FetchContent_Declare(
  dora_src
  GIT_REPOSITORY https://github.com/dora-rs/dora.git
  GIT_TAG        v0.3.4
)
FetchContent_MakeAvailable(dora_src)

# Path to the C API crate
set(DORA_C_API_PATH "${dora_src_SOURCE_DIR}/apis/c/node")

# Define a custom target to build the library using Cargo
# This produces 'libdora_node_api_c.a' in target/release
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/libdora_node_api_c.a"
    COMMAND cargo build --release --manifest-path ${DORA_C_API_PATH}/Cargo.toml --target-dir ${CMAKE_CURRENT_BINARY_DIR}/cargo-target
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/cargo-target/release/libdora_node_api_c.a ${CMAKE_CURRENT_BINARY_DIR}/libdora_node_api_c.a
    DEPENDS ${DORA_C_API_PATH}/src/lib.rs ${DORA_C_API_PATH}/Cargo.toml
    WORKING_DIRECTORY ${dora_src_SOURCE_DIR}
    COMMENT "Building Dora C API with Cargo..."
)

add_custom_target(dora_c_lib DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/libdora_node_api_c.a")

add_library(dora::node_api UNKNOWN IMPORTED)
set_target_properties(dora::node_api PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/libdora_node_api_c.a"
    INTERFACE_INCLUDE_DIRECTORIES "${DORA_C_API_PATH}"
)
add_dependencies(dora::node_api dora_c_lib)

# -----------------------------------------------------------------------------
# Main Executable
# -----------------------------------------------------------------------------
# Note: Linking MacOS frameworks (CoreFoundation, SystemConfiguration) required by Rust/dora
if (APPLE)
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    find_library(SYSTEMCONFIGURATION_LIBRARY SystemConfiguration)
    find_library(SECURITY_LIBRARY Security)
endif()

# We use the real main.cpp, not the stubs
add_executable(cpp-controller cpp/main.cpp)
target_link_libraries(cpp-controller PRIVATE dora::node_api)

if(APPLE)
    target_link_libraries(cpp-controller PRIVATE 
        ${COREFOUNDATION_LIBRARY} 
        ${SYSTEMCONFIGURATION_LIBRARY}
        ${SECURITY_LIBRARY}
    )
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(cpp-controller PRIVATE pthread dl)
endif()

