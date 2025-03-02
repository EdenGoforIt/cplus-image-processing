
#########################################################
# Source Files
#########################################################

SET(sources	"camera_with_fps.cpp")
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.11.0")



add_executable(camera_with_fps ${sources})

# Link with OpenCV Library
target_link_libraries(camera_with_fps PRIVATE ${OpenCV_LIBS})