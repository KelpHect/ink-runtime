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
  src/io/tokenizer.cpp
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
  # volk needs Vulkan headers on the include path: prefer an explicit SDK,
  # else the env var, else the known local install.
  if(NOT VULKAN_HEADERS_INSTALL_DIR)
    if(DEFINED ENV{VULKAN_SDK} AND EXISTS "$ENV{VULKAN_SDK}/Include/vulkan/vulkan.h")
      set(VULKAN_HEADERS_INSTALL_DIR "$ENV{VULKAN_SDK}" CACHE PATH "Vulkan SDK root")
    elseif(EXISTS "C:/VulkanSDK/1.4.363.0/Include/vulkan/vulkan.h")
      set(VULKAN_HEADERS_INSTALL_DIR "C:/VulkanSDK/1.4.363.0" CACHE PATH "Vulkan SDK root")
    endif()
  endif()
  set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.363.0)
  FetchContent_MakeAvailable(volk)
  target_compile_definitions(inkcore PUBLIC INK_WITH_VULKAN=1 VK_NO_PROTOTYPES)
  target_link_libraries(inkcore PUBLIC volk)
  target_sources(inkcore PRIVATE
    src/vk/vk.cpp
    src/vk/mem.cpp
    src/vk/cmd.cpp
    src/vk/pipe.cpp
    src/gpu/vk_backend.cpp
    src/gpu/gpu_graph.cpp
    src/gpu/gpu_prefill.cpp)
  target_include_directories(inkcore PUBLIC ${CMAKE_SOURCE_DIR}/shaders)

  # Compile shaders/*.comp -> ${CMAKE_BINARY_DIR}/shaders/<name>.spv.
  # shaders/ may not exist yet; the glob tolerates empty.
  find_program(INK_GLSLC glslc
    HINTS "C:/VulkanSDK/1.4.363.0/Bin" "$ENV{VULKAN_SDK}/Bin")
  file(GLOB INK_SHADER_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/shaders/*.comp")
  set(INK_SPVS "")
  if(INK_GLSLC)
    foreach(src ${INK_SHADER_SOURCES})
      get_filename_component(name ${src} NAME_WE)
      set(spv ${CMAKE_BINARY_DIR}/shaders/${name}.spv)
      add_custom_command(OUTPUT ${spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
        COMMAND ${INK_GLSLC} -O --target-env=vulkan1.4 -o ${spv} ${src}
        DEPENDS ${src}
        COMMENT "glslc ${name}.comp")
      list(APPEND INK_SPVS ${spv})
    endforeach()
  else()
    message(WARNING "glslc not found; .comp shaders will not be compiled")
  endif()
  add_custom_target(ink_shaders DEPENDS ${INK_SPVS})
  add_dependencies(inkcore ink_shaders)
endif()

add_executable(ink src/main.cpp)
target_link_libraries(ink PRIVATE inkcore)
if(INK_WITH_VULKAN)
  target_compile_definitions(ink PRIVATE INK_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")
endif()
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

  add_executable(ink_tokenizer tests/test_tokenizer.cpp)
  target_link_libraries(ink_tokenizer PRIVATE inkcore)
  target_compile_definitions(ink_tokenizer PRIVATE INK_MODEL_DIR="${CMAKE_SOURCE_DIR}/Inkling-2-Mini")
  if(MSVC)
    target_compile_options(ink_tokenizer PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
    target_compile_definitions(ink_tokenizer PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
  endif()
  add_test(NAME tokenizer COMMAND ink_tokenizer)
  if(INK_WITH_VULKAN)
    add_executable(test_vk_smoke tests/test_vk_smoke.cpp)
    target_link_libraries(test_vk_smoke PRIVATE inkcore)
    target_compile_definitions(test_vk_smoke PRIVATE
      INK_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders")
    if(MSVC)
      target_compile_options(test_vk_smoke PRIVATE /W4 /permissive- /Zc:__cplusplus /utf-8)
      target_compile_definitions(test_vk_smoke PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    endif()
    add_test(NAME vk_smoke COMMAND test_vk_smoke)
  endif()
endif()
