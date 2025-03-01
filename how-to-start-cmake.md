# Getting Started with CMake

## Prerequisites

- CMake (install via `brew install cmake` on macOS)
- A C++ compiler (clang++ comes with Xcode Command Line Tools)
- Visual Studio Code with C++ and CMake extensions

## Project Structure

Create a basic project structure:

```bash
mkdir my_project
cd my_project
mkdir src include build
```

## Create Initial Files

### 1. CMakeLists.txt

Create the main CMake configuration file in the root directory:

```cmake
cmake_minimum_required(VERSION 3.11)
project("159731_Project" CXX C)

# Enable IDE Project Folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#########################################################
# Find OpenCV
#########################################################
find_package(OpenCV REQUIRED)

#########################################################
# Include directories
#########################################################
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/include")

# ... rest of CMake configuration ...
```

### 2. Main Source File

Create a new file `src/main.cpp`:

```cpp
#include <iostream>

int main() {
    std::cout << "Hello, CMake!" << std::endl;
    return 0;
}
```

Create `src/CMakeLists.txt`:

```cmake
#########################################################
# Source Files
#########################################################
SET(sources "main.cpp")

add_executable(main ${sources})

# Link with OpenCV Library
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
```

## Build and Run

1. Configure the project:

```bash
mkdir build && cd build
cmake ..
```

2. Build the project:

```bash
make
```

3. Run the executable:

```bash
./src/main
```

## VS Code Integration

1. Install these VS Code extensions:

   - C/C++
   - CMake
   - CMake Tools

2. Create `.vscode/settings.json`:

```json
{
  "cmake.configureOnOpen": true,
  "cmake.buildDirectory": "${workspaceFolder}/build"
}
```

## Common CMake Commands

- `cmake ..` - Configure project
- `cmake --build .` - Build project
- `cmake --build . --clean-first` - Clean and build
- `cmake --build . --target clean` - Clean build files

## Debugging

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/main",
      "args": [],
      "stopAtEntry": false,
      "cwd": "${workspaceFolder}",
      "environment": [],
      "externalConsole": false,
      "MIMode": "lldb"
    }
  ]
}
```

Now you can press F5 to debug your program in VS Code.
