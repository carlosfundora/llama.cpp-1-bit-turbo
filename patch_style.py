import sys

def process_file():
    with open('gguf-py/tests/test_gguf_reader.py', 'r') as f:
        content = f.read()

    # The flake8 errors say we need 2 blank lines before functions, and we have 1.
    content = content.replace("class TestGGUFReaderArray(unittest.TestCase):", "\nclass TestGGUFReaderArray(unittest.TestCase):")
    content = content.replace("if __name__ == '__main__':", "\nif __name__ == '__main__':")

    with open('gguf-py/tests/test_gguf_reader.py', 'w') as f:
        f.write(content)

process_file()
