import sys
import re

file_path = "common/ngram-mod.cpp"
with open(file_path, "r") as f:
    content = f.read()

# Replace __builtin_prefetch
# If using _mm_prefetch, we need <immintrin.h> or <intrin.h> for Windows MSVC

# Add missing include for Windows if not present
if "<intrin.h>" not in content:
    content = content.replace("#include <string>\n", "#include <string>\n#if defined(_MSC_VER)\n#include <intrin.h>\n#endif\n")

# Replace __builtin_prefetch with something like:
# #if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
# _mm_prefetch((const char *) ..., _MM_HINT_T0);
# #elif defined(__GNUC__) || defined(__clang__)
# __builtin_prefetch(...);
# #endif

# Actually, the instructions explicitly said:
# "To maintain cross-platform C/C++ compatibility, particularly for Windows MSVC, use `<intrin.h>` and `_mm_prefetch` instead of compiler-specific intrinsics like `__builtin_prefetch`."
# It does not ask to just use _mm_prefetch everywhere, it says use it "instead of compiler-specific intrinsics like __builtin_prefetch" for MSVC compatibility.

content = content.replace("__builtin_prefetch", "prefetch_for_read")

# Now add a prefetch_for_read macro at the top

macro = """
#if defined(_MSC_VER) && (defined(_M_AMD64) || defined(_M_IX86))
#include <intrin.h>
#define prefetch_for_read(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
#define prefetch_for_read(addr) __builtin_prefetch(addr)
#else
#define prefetch_for_read(addr)
#endif
"""

if "prefetch_for_read" not in content:
    content = content.replace("#include <vector>\n", "#include <vector>\n" + macro + "\n")
elif "prefetch_for_read(addr)" not in content:
    # Just replace it properly
    # wait
    pass

with open(file_path, "w") as f:
    f.write(content)
