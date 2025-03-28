## How to Build and Run the Project

### Prerequisites

- CMake (3.11 or higher) - Build system generator
- OpenCV - Computer vision library
- A C++ compiler (e.g., g++) - Code compiler

### Build Instructions

1. If downloaded from the lecture (Using linux)

   - Delete build file and start over

2. Create a build directory and navigate into it:

   ```sh
   mkdir build
   cd build
   ```

   This creates a separate directory for build artifacts to keep source code clean.

3. Run CMake to configure the project:

   ```sh
   cmake ..
   ```

   CMake will:

   - Read the CMakeLists.txt files
   - Check for required dependencies (OpenCV)
   - Detect system architecture (ARM64/x86)
   - Generate build files (Makefiles)
   - Configure compiler settings and flags

4. Build the project:
   ```sh
   make
   ```
   Make will:
   - Read the generated Makefiles
   - Compile source files into object files
   - Link object files with libraries (OpenCV)
   - Create the final executable
   - Place the executable in build/src/camera_with_fps

### Run Instructions

1. After building, you can run the executable from the build directory:
   In the build folder, execute

   ```sh
   ./src/camera_with_fps
   ```

2. Ensure your camera is connected and properly configured.

3. Follow any on-screen instructions provided by the application.

# Others

```
/opt/homebrew/include/opencv4/opencv2


 g++ camera_with_fps.cpp  -o cv -I/opt/homebrew/include/opencv4/opencv2
```

Actually the open cv can be found here

```
-- Found OpenCV: /opt/homebrew/Cellar/opencv/4.11.0 (found version "4.11.0")

```

## Holy shit!!! How to make sure that intellisense is working for open cv.

1. Create a folder called include
2. Copy paste include folder from [text](../../../../../opt/homebrew/Cellar/opencv/4.11.0/include/opencv4)
3. rm -rf build && mkdir build && cd build
4. cmake ..
5. make
