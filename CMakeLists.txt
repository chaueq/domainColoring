cmake_minimum_required(VERSION 3.2)
project(domainColoring LANGUAGES CXX CUDA)
set( CMAKE_CXX_FLAGS "-pthread -std=c++17 -O3")
find_package( OpenCV REQUIRED )
find_package(CUDA REQUIRED)

file(GLOB SRC_domainColoring ./src/domainColoring/*)
add_executable(domainColoring ${SRC_domainColoring})
target_link_libraries(domainColoring ${OpenCV_LIBS})

file(GLOB SRC_compressor ./src/compressor/*)
add_executable(compressor ${SRC_compressor})
target_link_libraries(compressor ${OpenCV_LIBS})

file(GLOB SRC_player ./src/player/*)
add_executable(player ${SRC_player})
target_link_libraries(player ${OpenCV_LIBS})
