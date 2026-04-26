The HIP compiler isn't correctly configured in the testing environment. ROCm isn't installed in the environment where the agent is running (`Failed to find ROCm root directory`).
Because of this, I can't build tests.
However, I can still verify `ctest` without HIP or `cmake --build build` on the host to ensure `ggml-cuda.cu` didn't break things that would be caught by general C++ analysis, although CUDA/HIP specific code won't compile without the respective SDK.
Actually, it fails at configure time. I'll just skip the `ctest` part since I'm restricted by the environment and write the Journal.
