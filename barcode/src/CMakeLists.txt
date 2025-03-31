cmake_minimum_required(VERSION 3.11)
project(Chapter2)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV path for M2 Mac
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.11.0")
find_package(OpenCV REQUIRED)

#########################################################
# Source Files
#########################################################
# Find all cpp files in current directory
file(GLOB CPP_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# Create executable for each cpp file
foreach(source_file ${CPP_SOURCES})
    # Get filename without extension
    get_filename_component(filename ${source_file} NAME_WE)
    
    # Add executable
    add_executable(${filename} ${source_file})
    
    # Link OpenCV to each executable
    target_link_libraries(${filename} PRIVATE ${OpenCV_LIBS})
endforeach()