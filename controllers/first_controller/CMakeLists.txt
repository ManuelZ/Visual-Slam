cmake_minimum_required(VERSION 3.0)

# Setup the project.
# Its name is defined to be the controller directory name.
get_filename_component(PROJECT ${CMAKE_SOURCE_DIR} NAME)
project(${PROJECT})

# Get C or C++ sources in the current directory (only).
file(GLOB C_SOURCES *.c)
file(GLOB CPP_SOURCES *.cpp)
set(SOURCES ${C_SOURCES} ${CPP_SOURCES})

# Set the  Webots home path (change it according to your installation method)
set(WEBOTS_HOME "/usr/local/webots")

# Link with the Webots controller library
link_directories($ENV{WEBOTS_HOME}/lib/controller)

set (
  LIBRARIES 
  m ${CMAKE_SHARED_LIBRARY_PREFIX}Controller${CMAKE_SHARED_LIBRARY_SUFFIX}
  ${CMAKE_SHARED_LIBRARY_PREFIX}CppController${CMAKE_SHARED_LIBRARY_SUFFIX}
)

include_directories(
  $ENV{WEBOTS_HOME}/include/controller/c 
  $ENV{WEBOTS_HOME}/include/controller/cpp  
)

# Add Opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add Eigen header files
include_directories("/usr/include/eigen3")

# Add a library to the project using the specified source files
# add_executable(useHello useHello.cpp)
# target_link_libraries(useHello hello_shared)

# Setup the target executable
add_executable(${PROJECT} ${SOURCES})
target_link_libraries(${PROJECT} ${LIBRARIES})
target_link_libraries(${PROJECT} ${OpenCV_LIBS})

# Echo variables
message(ENV="${ENV}")
message(WEBOTS_HOME="${WEBOTS_HOME}")
message(PROJECT="${PROJECT}")
message(SOURCES="${SOURCES}")
message(LIBRARIES="${LIBRARIES}")


# Copy the target executable at the right location.
add_custom_command(TARGET ${PROJECT} POST_BUILD COMMAND ${CMAKE_COMMAND} -E
  copy ${CMAKE_BINARY_DIR}/${PROJECT} ${CMAKE_SOURCE_DIR}
)