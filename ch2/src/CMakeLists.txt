
#########################################################
# Source Files
#########################################################

SET(sources	"main.cpp")
set(OpenCV_DIR "/opt/homebrew/Cellar/opencv/4.11.0")



add_executable(main ${sources})

# Link with OpenCV Library
target_link_libraries(main PRIVATE ${OpenCV_LIBS})