cmake_minimum_required(VERSION 3.15)
project(KMeansClustering)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-O3")

find_package(Eigen3 REQUIRED)

add_executable(kmeans
    src/main.cpp
    src/kmeans.cpp
    src/dataset.cpp
)

target_link_libraries(kmeans Eigen3::Eigen)
