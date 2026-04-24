🎯 **What:**
Fixed insecure deserialization vulnerabilities in `tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py`.

⚠️ **Risk:**
Using `torch.load` without `weights_only=True` is insecure because it relies on `pickle` for deserialization. Maliciously crafted `.clip` or projector files could lead to arbitrary code execution when loaded, compromising the system running the conversion script.

🛡️ **Solution:**
Updated both instances of `torch.load` in `minicpmv-convert-image-encoder-to-gguf.py` to include `weights_only=True`, restricting deserialization to safe data structures and preventing arbitrary code execution.
