cmake_minimum_required(VERSION 3.24)
project(ink-runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# First milestone builds with NO GPU deps: Vulkan stays off until the backend lands.
option(INK_WITH_VULKAN "Vulkan compute backend" OFF)
option(INK_BUILD_TESTS "Build smoke tests" ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

add_library(inkcore STATIC
  src/io/safetensors.cpp
  src/io/config.cpp
  src/model/ops.cpp
  src/model/graph.cpp)
target_include_directories(inkcore PUBLIC include src)
target_link_libraries(inkcore PUBLIC nlohmann_json::nlohmann_json)

if(MSVC)
  target_compile_options(inkcore PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8 /bigobj)
  target_compile_definitions(inkcore PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  target_compile_options(inkcore PRIVATE -Wall -Wextra)
endif()

if(INK_WITH_VULKAN)
  FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG 1.4.304)
  FetchContent_MakeAvailable(volk)
  target_compile_definitions(inkcore PUBLIC INK_WITH_VULKAN=1 VK_NO_PROTOTYPES)
  target_link_libraries(inkcore PUBLIC volk)
endif()

add_executable(ink src/main.cpp)
target_link_libraries(ink PRIVATE inkcore)
if(MSVC)
  target_compile_options(ink PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
  target_compile_definitions(ink PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

if(INK_BUILD_TESTS)
  enable_testing()
  add_executable(ink_smoke tests/test_smoke.cpp)
  target_link_libraries(ink_smoke PRIVATE inkcore)
  if(MSVC)
    target_compile_options(ink_smoke PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
    target_compile_definitions(ink_smoke PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  add_test(NAME smoke COMMAND ink_smoke)
endif()
