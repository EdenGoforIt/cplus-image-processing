
#########################################################
# Source Files
#########################################################

SET(sources	"camera_with_fps.cpp")

add_executable(camera_with_fps ${sources})

# Link with OpenCV Library
target_link_libraries(camera_with_fps PRIVATE ${OpenCV_LIBS})