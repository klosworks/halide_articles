cmake_minimum_required(VERSION 3.16)
project(HalideExample)

set(CMAKE_CXX_STANDARD 17)  # or newer
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(Halide REQUIRED)
find_package(OpenMP)

add_executable(my_generators value_spread_generator.cpp)
target_link_libraries(my_generators PRIVATE Halide::Generator)
add_halide_library(value_spread_lib FROM my_generators GENERATOR value_spread_generator)

add_executable(main main_ellipse.cpp)
target_link_libraries(main PRIVATE Halide::Halide)
target_link_libraries(main PUBLIC ${OpenCV_LIBS} )

add_executable(value_spread value_spread.cpp)
target_link_libraries(value_spread PRIVATE Halide::Halide)
target_link_libraries(value_spread PUBLIC ${OpenCV_LIBS} )
target_link_libraries(value_spread PRIVATE value_spread_lib)

if(OpenMP_CXX_FOUND)
    target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(value_spread PUBLIC OpenMP::OpenMP_CXX)
endif()
