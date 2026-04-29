import unittest
import tempfile
import os
import struct

from gguf.gguf_reader import GGUFReader
from gguf.constants import GGUFValueType, GGUF_MAGIC


class TestGGUFReaderArray(unittest.TestCase):
    def setUp(self):
        self.fd, self.path = tempfile.mkstemp()

    def tearDown(self):
        os.close(self.fd)
        os.remove(self.path)

    def write_gguf_header(self, f):
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', 3))
        f.write(struct.pack('<Q', 0))
        f.write(struct.pack('<Q', 1))

    def write_string(self, f, s):
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def test_multi_dimensional_array(self):
        with open(self.path, 'wb') as f:
            self.write_gguf_header(f)
            self.write_string(f, 'test.array')
            f.write(struct.pack('<I', GGUFValueType.ARRAY.value))

            f.write(struct.pack('<I', GGUFValueType.ARRAY.value))
            f.write(struct.pack('<Q', 2))

            f.write(struct.pack('<I', GGUFValueType.INT32.value))
            f.write(struct.pack('<Q', 2))
            f.write(struct.pack('<i', 10))
            f.write(struct.pack('<i', 20))

            f.write(struct.pack('<I', GGUFValueType.INT32.value))
            f.write(struct.pack('<Q', 3))
            f.write(struct.pack('<i', 30))
            f.write(struct.pack('<i', 40))
            f.write(struct.pack('<i', 50))

            padding = f.tell() % 32
            if padding != 0:
                f.write(b'\x00' * (32 - padding))

        reader = GGUFReader(self.path)
        field = reader.get_field('test.array')
        self.assertIsNotNone(field)

        assert field is not None
        contents = field.contents()
        self.assertEqual(contents, [[10, 20], [30, 40, 50]])

        assert field is not None
        self.assertEqual(field.contents(0), [10, 20])
        assert field is not None
        self.assertEqual(field.contents(1), [30, 40, 50])

    def test_multi_dimensional_array_strings(self):
        with open(self.path, 'wb') as f:
            self.write_gguf_header(f)
            self.write_string(f, 'test.string.array')
            f.write(struct.pack('<I', GGUFValueType.ARRAY.value))

            f.write(struct.pack('<I', GGUFValueType.ARRAY.value))
            f.write(struct.pack('<Q', 2))

            f.write(struct.pack('<I', GGUFValueType.STRING.value))
            f.write(struct.pack('<Q', 2))
            self.write_string(f, "hello")
            self.write_string(f, "world")

            f.write(struct.pack('<I', GGUFValueType.STRING.value))
            f.write(struct.pack('<Q', 1))
            self.write_string(f, "!")

            padding = f.tell() % 32
            if padding != 0:
                f.write(b'\x00' * (32 - padding))

        reader = GGUFReader(self.path)
        field = reader.get_field('test.string.array')
        self.assertIsNotNone(field)

        assert field is not None
        contents = field.contents()
        self.assertEqual(contents, [['hello', 'world'], ['!']])


if __name__ == '__main__':
    unittest.main()
