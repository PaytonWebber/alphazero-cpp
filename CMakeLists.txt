cmake_minimum_required(VERSION 3.10)
project(AlphaZero)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "libs/libtorch")

find_package(Torch REQUIRED)
include_directories(include src)

add_executable(main src/main.cpp)

target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 17)
