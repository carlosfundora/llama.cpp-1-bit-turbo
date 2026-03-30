<!--
@file: docs/subsystems/snapdragon-backend.md
@doc_type: subsystem_reference
@version: 1.0.0
@title: Snapdragon Backend
@summary: Canonical overview and reference for using llama.cpp on Qualcomm Snapdragon platforms.
@tags: [backend, snapdragon, hexagon, adreno, qualcomm]
@author: Registrar Prime
@copyright: © 2026 Carlos Fundora
@status: active
@last_updated: 2026-03-29
@changelog:
- 2026-03-29 [@Registrar Prime]: Restored factual cmake build steps and technical logs.
-->
# Snapdragon Backend

This document describes how to build and run llama.cpp on Qualcomm Snapdragon platforms.

The primary target platform is Windows on Snapdragon (WoS). However, because Android NDK shares the same Clang compiler, it is also very easy to build for Android.

llama.cpp supports three backends on Snapdragon-based devices:
1.  **CPU**
2.  **Adreno GPU** (via OpenCL)
3.  **Hexagon NPU** (HTP0-4)

The Hexagon NPU operates as a "GPU" device when configuring offload parameters like `-ngl`. You can select which backend to run the model on using the `--device` option (often mapped to a script variable like `D=`).

## Building

Building for Snapdragon is supported on Windows x86/x64/ARM64 and Linux x64/ARM64. The builds use a cross-compiling toolchain. Note that native compilation is not supported due to missing Hexagon NPU toolchains.

The repository includes a CMake preset for Snapdragon builds. The preset file is located at `docs/subsystems/snapdragon/CMakeUserPresets.json`. Note: This file was previously at `docs/backend/snapdragon/CMakeUserPresets.json`. To use it, copy it to the root of your `llama.cpp` tree:

```bash
cp docs/subsystems/snapdragon/CMakeUserPresets.json .
```

### Prerequisites

You need Docker, `git`, and `wget` installed on your host system.

### Option 1: Using provided Toolchain Docker Image (Recommended)

The easiest way to build is using the provided Docker toolchain image:

```bash
~/src/llama.cpp$ docker run --rm -it -v $(pwd):/app -w /app \
  ghcr.io/snapdragon-toolchain/snapdragon-toolchain:latest \
  bash -c "cmake --preset arm64-android-snapdragon-release -B build-android && cmake --build build-android && cmake --install build-android --prefix pkg-snapdragon"
...
~/src/llama.cpp$ ls -l pkg-snapdragon/
total 16
drwxr-xr-x 2 user group 4096 Oct 11 11:27 bin
drwxr-xr-x 2 user group 4096 Oct 11 11:27 include
drwxr-xr-x 2 user group 4096 Oct 11 11:27 lib
```

To build for Windows on Snapdragon, use the Windows preset:

```bash
~/src/llama.cpp$ docker run --rm -it -v $(pwd):/app -w /app \
  ghcr.io/snapdragon-toolchain/snapdragon-toolchain:latest \
  bash -c "cmake --preset arm64-windows-snapdragon-release -B build-windows && cmake --build build-windows && cmake --install build-windows --prefix pkg-snapdragon"
```

### Option 2: Build natively on Windows on Snapdragon

The build can also be run directly on a Snapdragon PC. However, setting this up takes a few manual steps. Please refer to [Snapdragon Windows Setup](../operations/snapdragon-windows-setup.md) for detailed instructions on installing the required SDKs and drivers, creating certificates, and test-signing the HTP ops libraries.

## How to Run

### Windows

If you built for Windows, all artifacts are installed in the `pkg-snapdragon` folder. Use the Powershell scripts in `scripts/snapdragon/windows` to run the tools with properly configured environment variables.

### Android

If you built for Android, use ADB (Android Debug Bridge) to push the binaries and models to the device. Note: Do this on the host, as the toolchain Docker image does not have ADB.

1.  Enable Developer Options and USB Debugging on your device.
2.  Push binaries:
    ```bash
    ~/src/llama.cpp$ adb push pkg-snapdragon/llama.cpp /data/local/tmp/
    ```
3.  Push a model:
    ```bash
    ~/src/llama.cpp$ wget https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
    ~/src/llama.cpp$ adb push Llama-3.2-1B-Instruct-Q4_0.gguf /data/local/tmp/gguf
    ```
4.  Run using the provided ADB wrapper scripts:
    ```bash
    ~/src/llama.cpp$ M=Llama-3.2-1B-Instruct-Q4_0.gguf D=HTP0 ./scripts/snapdragon/adb/run-completion.sh -p "what is the most popular cookie in the world?"
    ```

    Example output:
    ```
    ...
    ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
    ggml-hex: Hexagon Arch version v79
    ggml-hex: allocating new session: HTP0
    ggml-hex: new session: HTP0 : session-id 0 domain-id 3 uri file:///libggml-htp-v79.so?htp_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp&_session=0 handle 0xb4000072c7955e50
    ...
    load_tensors: offloading output layer to GPU
    load_tensors: offloaded 17/17 layers to GPU
    load_tensors:          CPU model buffer size =   225.49 MiB
    load_tensors:         HTP0 model buffer size =     0.26 MiB
    load_tensors:  HTP0-REPACK model buffer size =   504.00 MiB
    ...
    llama_memory_breakdown_print: | memory breakdown [MiB] | total   free    self   model   context   compute    unaccounted |
    llama_memory_breakdown_print: |   - HTP0 (Hexagon)     |  2048 = 2048 + (   0 =     0 +       0 +       0) +           0 |
    llama_memory_breakdown_print: |   - Host               |                  439 =   225 +     136 +      77                |
    llama_memory_breakdown_print: |   - HTP0-REPACK        |                  504 =   504 +       0 +       0                |
    ```

## Environment Variables

The Snapdragon Hexagon backend uses the following environment variables:

-   `GGML_HEXAGON_NDEV`: Controls the number of devices/sessions to allocate (default: 1). Large models (e.g., >4B parameters) require multiple sessions (2 for ~8B, 4 for ~20B).
-   `GGML_HEXAGON_NHVX`: Number of HVX hardware threads to use (default: 0, meaning all).
-   `GGML_HEXAGON_HOSTBUF`: Controls whether the backend allocates host buffers (default: 1). Required for ops needing REPACK buffers.
-   `GGML_HEXAGON_EXPERIMENTAL`: Enables experimental features (default: 1).
-   `GGML_HEXAGON_VERBOSE`: Enables verbose logging of Ops from the backend (default: 1).
-   `GGML_HEXAGON_PROFILE`: Generates a host-side profile.
-   `GGML_HEXAGON_OPMASK`: Allows enabling specific stages of the processing pipeline (`0x1` Queue, `0x2` Dynamic Quantizer, `0x4` Compute).

## Related Documents

-   [Developer Details & Architecture Notes](../engineering-notes/snapdragon-developer-details.md)
-   [Windows Setup & Test Signing](../operations/snapdragon-windows-setup.md)
