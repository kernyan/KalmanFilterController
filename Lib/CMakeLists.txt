file(GLOB SRCS ${PROJECT_SOURCE_DIR}/Lib/src/*.cpp)
#include_directories(${PROJECT_SOURCE_DIR}/main/)
add_library(Lib STATIC ${SRCS})

