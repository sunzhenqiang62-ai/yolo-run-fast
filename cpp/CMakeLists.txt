cmake_minimum_required(VERSION 3.14)
project(aerial_obb_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
endif()

find_package(Threads REQUIRED)
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# --- OpenCV ---
if(WIN32)
  set(OPENCV_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/opencv"
      CACHE PATH "Prebuilt OpenCV root (opencv/build/...)")
  set(OpenCV_DIR "${OPENCV_ROOT}/build/x64/vc16/lib")
  find_package(OpenCV REQUIRED)
else()
  find_package(OpenCV REQUIRED)
endif()

# --- ONNX Runtime ---
set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include"
    CACHE PATH "ONNX Runtime C++ headers")

if(WIN32)
  set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-x64"
      CACHE PATH "ONNX Runtime Windows SDK root")
  if(NOT EXISTS "${ORT_ROOT}/lib/onnxruntime.lib")
    message(FATAL_ERROR "onnxruntime.lib missing — run cpp/setup_deps.ps1")
  endif()
  set(ORT_LIBRARY "${ORT_ROOT}/lib/onnxruntime.lib")
  set(ORT_DLL "${ORT_ROOT}/lib/onnxruntime.dll")
  set(ORT_INCLUDE_DIR "${ORT_ROOT}/include")
else()
  set(ORT_LIB_DIR "${REPO_ROOT}/.venv/lib/python3.8/site-packages/onnxruntime/capi"
      CACHE PATH "Directory containing libonnxruntime.so")
  file(GLOB ORT_SO "${ORT_LIB_DIR}/libonnxruntime.so*")
  if(NOT ORT_SO)
    message(FATAL_ERROR "libonnxruntime.so not found in ${ORT_LIB_DIR}")
  endif()
  list(GET ORT_SO 0 ORT_LIBRARY)
endif()

message(STATUS "OpenCV: ${OpenCV_VERSION} @ ${OpenCV_DIR}")
message(STATUS "ONNX Runtime lib: ${ORT_LIBRARY}")
message(STATUS "ONNX Runtime headers: ${ORT_INCLUDE_DIR}")

add_executable(aerial_obb src/main.cpp)
target_include_directories(aerial_obb PRIVATE ${ORT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(aerial_obb PRIVATE ${ORT_LIBRARY} ${OpenCV_LIBS} Threads::Threads)

if(WIN32)
  add_custom_command(TARGET aerial_obb POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_DLL}" "$<TARGET_FILE_DIR:aerial_obb>/onnxruntime.dll"
    COMMENT "Copying onnxruntime.dll"
  )
  set(_ORT_SHARED "${ORT_ROOT}/lib/onnxruntime_providers_shared.dll")
  if(EXISTS "${_ORT_SHARED}")
    add_custom_command(TARGET aerial_obb POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${_ORT_SHARED}" "$<TARGET_FILE_DIR:aerial_obb>/onnxruntime_providers_shared.dll"
    )
  endif()
  # Prebuilt OpenCV ships opencv_world4xxx.dll beside the .lib.
  set(_OCV_DLL "${OpenCV_DIR}/../bin/opencv_world${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}.dll")
  if(EXISTS "${_OCV_DLL}")
    add_custom_command(TARGET aerial_obb POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${_OCV_DLL}" "$<TARGET_FILE_DIR:aerial_obb>"
      COMMENT "Copying OpenCV runtime DLL"
    )
  endif()
else()
  get_filename_component(ORT_SO_NAME "${ORT_LIBRARY}" NAME)
  add_custom_command(TARGET aerial_obb POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${ORT_LIBRARY}" "$<TARGET_FILE_DIR:aerial_obb>/libonnxruntime.so.1"
  )
  file(GLOB ORT_PROVIDER_SO "${ORT_LIB_DIR}/libonnxruntime_providers_*.so")
  foreach(prov ${ORT_PROVIDER_SO})
    get_filename_component(prov_name "${prov}" NAME)
    add_custom_command(TARGET aerial_obb POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "${prov}" "$<TARGET_FILE_DIR:aerial_obb>/${prov_name}"
    )
  endforeach()
  set(REPO_SITE_PACKAGES "${REPO_ROOT}/.venv/lib/python3.8/site-packages")
  file(GLOB ORT_CUDA_LIB_DIRS LIST_DIRECTORIES true "${REPO_SITE_PACKAGES}/nvidia/*/lib")
  set(ORT_RPATH "$ORIGIN" "${ORT_LIB_DIR}" ${ORT_CUDA_LIB_DIRS})
  set_target_properties(aerial_obb PROPERTIES BUILD_RPATH "${ORT_RPATH}" INSTALL_RPATH "${ORT_RPATH}")
  string(JOIN ":" ORT_RUNTIME_LIB_PATH "${ORT_LIB_DIR}" ${ORT_CUDA_LIB_DIRS})
  target_compile_definitions(aerial_obb PRIVATE ORT_RUNTIME_LIB_PATH="${ORT_RUNTIME_LIB_PATH}")
endif()
