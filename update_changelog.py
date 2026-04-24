import sys
from pathlib import Path

path = Path("CHANGELOG.md")
content = path.read_text() if path.exists() else ""

new_entry = """## Docs

- None required
"""

path.write_text(new_entry + content)
