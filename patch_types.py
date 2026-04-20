import sys

def process_file():
    with open('gguf-py/tests/test_gguf_reader.py', 'r') as f:
        content = f.read()

    content = content.replace("        if field is None:\n            self.fail(\"Field not found\")\n        assert field is not None\\n        contents = field.contents()", "        assert field is not None\n        contents = field.contents()")
    content = content.replace("        if field is None:\n            self.fail(\"Field not found\")\n        assert field is not None\\n        self.assertEqual(field.contents(0), [10, 20])", "        assert field is not None\n        self.assertEqual(field.contents(0), [10, 20])")
    content = content.replace("        assert field is not None\\n        self.assertEqual(field.contents(1), [30, 40, 50])", "        assert field is not None\n        self.assertEqual(field.contents(1), [30, 40, 50])")

    with open('gguf-py/tests/test_gguf_reader.py', 'w') as f:
        f.write(content)

process_file()
