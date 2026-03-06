cmake_minimum_required(VERSION 3.16)

project(strength)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED)
find_package(Boost COMPONENTS system thread iostreams)
find_package(ale REQUIRED)
find_package(OpenCV REQUIRED)

find_package(ZLIB REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZIP REQUIRED libzip)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -Wall -mpopcnt -O3 -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wno-unused-function -O0 -pthread")

# for git info
include_directories(${PROJECT_BINARY_DIR}/git_info)

add_subdirectory(extern/cnpy)

add_subdirectory(strength)
add_subdirectory(strength/misc)
add_subdirectory(strength/trainer)
add_subdirectory(minizero)

string(TOLOWER "${PROJECT_NAME}_${GAME_TYPE}" EXE_FILE_NAME)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${EXE_FILE_NAME})

target_link_libraries(strength PRIVATE cnpy ${ZLIB_LIBRARIES} ${LIBZIP_LIBRARIES})

# ===========================================================================
# USI Engine Extension (既存のビルド設定を一切壊さずに追記)
# ===========================================================================

add_executable(shogi_usi usi_main.cpp strength/misc/st_configuration.cpp)

target_link_libraries(shogi_usi
    PRIVATE
    ${TORCH_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${LIBZIP_LIBRARIES}
    cnpy
    # 既存のビルドで成功しているターゲットがリンクしているものを追加
    config
    network
    environment
    utils
    misc
)

target_include_directories(shogi_usi PRIVATE
    .
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/minizero
    ${CMAKE_SOURCE_DIR}/minizero/minizero                  # これが重要
    ${CMAKE_SOURCE_DIR}/minizero/minizero/configuration    # これも追加
    ${CMAKE_SOURCE_DIR}/minizero/minizero/environment/base # configuration.h用
    ${TORCH_INCLUDE_DIRS}
)

target_compile_options(shogi_usi PRIVATE -fPIC -g -Wall -mpopcnt -O3 -pthread)
set_target_properties(shogi_usi PROPERTIES CXX_STANDARD 17)

# ===========================================================================