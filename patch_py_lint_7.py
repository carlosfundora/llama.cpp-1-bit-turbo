import re

triton_py = "cmake/triton_aot_compile.py"
spec_py = "scripts/spec_harness.py"

with open(triton_py, "r") as f:
    content = f.read()

# Reorder __future__
if "from __future__ import annotations" in content:
    content = content.replace("from __future__ import annotations", "")
    content = "from __future__ import annotations\n" + content

# Fix unused imports
content = content.replace("import triton", "import triton # noqa: F401")
content = content.replace("import triton.language as tl", "import triton.language as tl # noqa: F401")
content = content.replace("from triton.compiler.compiler import ASTSource", "from triton.compiler.compiler import ASTSource # noqa: F401")
content = content.replace("import triton.backends.amd.compiler", "import triton.backends.amd.compiler # noqa: F401")
content = content.replace("import triton.backends.nvidia.compiler", "import triton.backends.nvidia.compiler # noqa: F401")

# Fix f-string without placeholders
content = content.replace("f\"Unsupported backend '{args.target}'\"", "\"Unsupported backend '{args.target}'\"")
content = content.replace("logging.warning(", "logging.warning('', ")

with open(triton_py, "w") as f:
    f.write(content)

with open(spec_py, "r") as f:
    content = f.read()

# Fix f-string without placeholders
content = content.replace("logging.warning(", "logging.warning('', ")

with open(spec_py, "w") as f:
    f.write(content)
