import re

files_to_fix = [
    "convert_hf_to_gguf.py",
    "examples/model-conversion/scripts/causal/compare-logits.py",
    "examples/model-conversion/scripts/utils/check-nmse.py",
    "examples/model-conversion/scripts/utils/compare_tokens.py",
    "examples/model-conversion/scripts/utils/semantic_check.py",
    "gguf-py/gguf/lazy.py",
    "gguf-py/gguf/vocab.py",
    "scripts/jinja/jinja-tester.py",
    "scripts/server-bench.py"
]

for file in files_to_fix:
    with open(file, "r") as f:
        content = f.read()

    # Simple regex to remove type: ignore and ty: ignore
    content = re.sub(r'#\s*type:\s*ignore.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'#\s*ty:\s*ignore.*$', '', content, flags=re.MULTILINE)

    # Clean up trailing spaces caused by removal
    content = re.sub(r' \n', '\n', content)

    with open(file, "w") as f:
        f.write(content)

print("Removed ty: ignore warnings")
