file(GLOB SRCS *.cpp)

add_library(misc ${SRCS})
target_include_directories(
    misc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/extern/cnpy
)
target_link_libraries(
    misc
    PRIVATE
    actor
    config
    console
    environment
    network
    utils
    cnpy
)