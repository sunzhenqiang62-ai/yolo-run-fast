cmake_minimum_required(VERSION 3.14)
project(aerial_obb_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# ONNX Runtime: headers pulled into third_party/ by setup_onnxruntime_headers.sh;
# the shared library ships inside the project venv (we do NOT modify the venv).
set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")

# Allow overriding the venv location; default to the repo's .venv.
get_filename_component(REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(ORT_LIB_DIR "${REPO_ROOT}/.venv/lib/python3.8/site-packages/onnxruntime/capi"
    CACHE PATH "Directory containing libonnxruntime.so")

file(GLOB ORT_SO "${ORT_LIB_DIR}/libonnxruntime.so*")
if(NOT ORT_SO)
  message(FATAL_ERROR "libonnxruntime.so not found in ${ORT_LIB_DIR}")
endif()
list(GET ORT_SO 0 ORT_LIBRARY)
message(STATUS "ONNX Runtime lib: ${ORT_LIBRARY}")
message(STATUS "ONNX Runtime headers: ${ORT_INCLUDE_DIR}")

add_executable(aerial_obb
  src/main.cpp
)
target_include_directories(aerial_obb PRIVATE
  ${ORT_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(aerial_obb PRIVATE
  ${ORT_LIBRARY}
  ${OpenCV_LIBS}
  Threads::Threads
)

# The venv ships libonnxruntime.so.1.19.2 but its SONAME is libonnxruntime.so.1
# and no such symlink exists. We must NOT modify the venv, so create the SONAME
# symlink in our build dir and point RPATH at both that dir ($ORIGIN) and the
# venv capi dir (for the CUDA/TensorRT provider .so files).
get_filename_component(ORT_SO_NAME "${ORT_LIBRARY}" NAME)
add_custom_command(TARGET aerial_obb POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
          "${ORT_LIBRARY}" "$<TARGET_FILE_DIR:aerial_obb>/libonnxruntime.so.1"
  COMMENT "Creating libonnxruntime.so.1 SONAME symlink -> ${ORT_SO_NAME}"
)

# ORT's provider bridge dlopens its sibling provider .so files relative to the
# loaded libonnxruntime.so. Because we load via the build-dir symlink, the
# CUDA/TensorRT provider libs must sit alongside it. Symlink them in too.
file(GLOB ORT_PROVIDER_SO "${ORT_LIB_DIR}/libonnxruntime_providers_*.so")
foreach(prov ${ORT_PROVIDER_SO})
  get_filename_component(prov_name "${prov}" NAME)
  add_custom_command(TARGET aerial_obb POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            "${prov}" "$<TARGET_FILE_DIR:aerial_obb>/${prov_name}"
    COMMENT "Linking ORT provider ${prov_name}"
  )
endforeach()

# The ORT-GPU build links against CUDA 12 / cuDNN 9, which the venv ships via
# the nvidia-*-cu12 pip packages (system CUDA here is 11). Add every
# nvidia/*/lib dir to RPATH so the CUDA provider's deps resolve at runtime.
set(REPO_SITE_PACKAGES "${REPO_ROOT}/.venv/lib/python3.8/site-packages")
file(GLOB ORT_CUDA_LIB_DIRS LIST_DIRECTORIES true "${REPO_SITE_PACKAGES}/nvidia/*/lib")
set(ORT_RPATH "$ORIGIN" "${ORT_LIB_DIR}" ${ORT_CUDA_LIB_DIRS})

set_target_properties(aerial_obb PROPERTIES
  BUILD_RPATH "${ORT_RPATH}"
  INSTALL_RPATH "${ORT_RPATH}"
)
# DT_RPATH/RUNPATH on the executable is not consulted by the loader when
# resolving a dlopen'd library's own dependencies (the ORT CUDA provider pulls
# in libcublasLt.so.12 etc. from the venv nvidia-*-cu12 packages). So the binary
# re-execs itself once with LD_LIBRARY_PATH set to these dirs (see main.cpp).
# Bake the path list in at compile time from the venv layout.
string(JOIN ":" ORT_RUNTIME_LIB_PATH "${ORT_LIB_DIR}" ${ORT_CUDA_LIB_DIRS})
target_compile_definitions(aerial_obb PRIVATE
  ORT_RUNTIME_LIB_PATH="${ORT_RUNTIME_LIB_PATH}")
