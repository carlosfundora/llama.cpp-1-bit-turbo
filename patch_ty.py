with open("ty.toml", "r") as f:
    content = f.read()

new_rule = """
unused-ignore-comment = "ignore"
unused-type-ignore-comment = "ignore"
"""
with open("ty.toml", "w") as f:
    f.write(content + "\n" + new_rule)
