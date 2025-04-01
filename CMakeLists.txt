cmake_minimum_required(VERSION 3.10)
project(AlphaZero)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "libs/libtorch")

find_package(Torch REQUIRED)
include_directories(include src)

# add_executable(main_test src/main_test.cpp)
add_executable(main_infer src/main_inference.cpp src/tictactoe.cpp)
add_executable(main_train src/main_train.cpp src/tictactoe.cpp)

target_link_libraries(main_infer "${TORCH_LIBRARIES}")
set_property(TARGET main_infer PROPERTY CXX_STANDARD 17)

target_link_libraries(main_train "${TORCH_LIBRARIES}")
set_property(TARGET main_train PROPERTY CXX_STANDARD 17)
