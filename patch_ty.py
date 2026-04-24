with open("ty.toml", "r") as f:
    content = f.read()

new_rule = """
[[overrides]]
include = [
    "**/*.py"
]
[overrides.rules]
unresolved-import = "ignore"
"""

with open("ty.toml", "w") as f:
    f.write(content.replace("[overrides.rules]\nunresolved-import = \"ignore\"", "") + "\n" + new_rule)
