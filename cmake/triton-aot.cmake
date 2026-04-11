# cmake/triton-aot.cmake
# Optional Triton AOT compilation of fused RotorQuant kernels.
#
# Enable with: cmake -DGGML_TRITON=ON
# Requires Triton 3.6+ in the Python environment at build time only.
# No Python dependency at inference time.

cmake_minimum_required(VERSION 3.18)

option(GGML_TRITON "Pre-compile fused RotorQuant Triton kernels at build time" OFF)

if (GGML_TRITON)
    message(STATUS "GGML_TRITON=ON: enabling Triton AOT kernel compilation")

    # Locate Python3 (build-time only)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    # Verify Triton is importable
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import triton; print(triton.__version__)"
        OUTPUT_VARIABLE _TRITON_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _TRITON_FOUND
    )
    if (NOT _TRITON_FOUND EQUAL 0)
        message(FATAL_ERROR
            "GGML_TRITON=ON but 'import triton' failed. "
            "Install Triton 3.6+: pip install triton")
    endif()
    message(STATUS "Triton version: ${_TRITON_VERSION}")

    # Determine target backend from GGML_USE_HIP / GGML_USE_CUDA
    if (DEFINED GGML_USE_HIP OR DEFINED ENV{ROCM_PATH})
        set(GGML_TRITON_BACKEND "hip")
        # Detect GPU arch from rocminfo if not explicitly set
        if (NOT DEFINED GGML_TRITON_ARCH)
            execute_process(
                COMMAND bash -c "rocminfo 2>/dev/null | awk '/Name:.*gfx/{print $2; exit}'"
                OUTPUT_VARIABLE _ROCM_ARCH
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _ROCM_ARCH_FOUND
            )
            if (_ROCM_ARCH_FOUND EQUAL 0 AND NOT "${_ROCM_ARCH}" STREQUAL "")
                set(GGML_TRITON_ARCH "${_ROCM_ARCH}")
            else()
                set(GGML_TRITON_ARCH "gfx1031")
            endif()
        endif()
    else()
        set(GGML_TRITON_BACKEND "cuda")
        if (NOT DEFINED GGML_TRITON_ARCH)
            set(GGML_TRITON_ARCH "sm_80")
        endif()
    endif()

    message(STATUS "Triton AOT target: ${GGML_TRITON_BACKEND}:${GGML_TRITON_ARCH}")

    # Output directory for compiled kernels
    set(GGML_TRITON_KERNEL_DIR "${CMAKE_BINARY_DIR}/triton-kernels" CACHE PATH
        "Directory where compiled Triton .hsaco/.cubin files are stored")

    set(_TRITON_COMPILE_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/triton_aot_compile.py")

    # Stamp file to track whether kernels are up-to-date
    set(_TRITON_STAMP "${GGML_TRITON_KERNEL_DIR}/.compiled.stamp")

    add_custom_command(
        OUTPUT  "${_TRITON_STAMP}"
        COMMAND ${Python3_EXECUTABLE} "${_TRITON_COMPILE_SCRIPT}"
                    --output-dir "${GGML_TRITON_KERNEL_DIR}"
                    --target     "${GGML_TRITON_BACKEND}"
                    --arch       "${GGML_TRITON_ARCH}"
        COMMAND ${CMAKE_COMMAND} -E touch "${_TRITON_STAMP}"
        DEPENDS "${_TRITON_COMPILE_SCRIPT}"
        COMMENT "Compiling fused RotorQuant Triton kernels (${GGML_TRITON_BACKEND}:${GGML_TRITON_ARCH})"
        VERBATIM
    )

    add_custom_target(triton_kernels ALL
        DEPENDS "${_TRITON_STAMP}"
    )

    # Propagate flags to C++ compilation
    add_compile_definitions(
        GGML_TRITON=1
        GGML_TRITON_KERNEL_DIR="${GGML_TRITON_KERNEL_DIR}"
    )

    message(STATUS "Triton kernels will be written to: ${GGML_TRITON_KERNEL_DIR}")
endif()
