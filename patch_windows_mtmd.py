import sys

with open('tools/mtmd/mtmd-audio.cpp', 'r') as f:
    content = f.read()

content = content.replace("std::fill(mels.begin(), mels.begin() + chunk.size(), 0.0);", "std::fill(mels.begin(), mels.begin() + chunk.size(), 0.0f);")
content = content.replace("std::fill(padded.begin() + mels.size(), padded.end(), 0.0);", "std::fill(padded.begin() + mels.size(), padded.end(), 0.0f);")
content = content.replace("std::fill(padded.begin() + mels.size(), padded.end(), 0);", "std::fill(padded.begin() + mels.size(), padded.end(), 0.0f);")
content = content.replace("std::fill(chunk.begin() + size, chunk.end(), 0);", "std::fill(chunk.begin() + size, chunk.end(), 0.0f);")

with open('tools/mtmd/mtmd-audio.cpp', 'w') as f:
    f.write(content)

with open('tools/mtmd/clip.cpp', 'r') as f:
    content = f.read()

content = content.replace("std::fill(res.begin() + image_size.width * image_size.height * 3, res.end(), 0.0);", "std::fill(res.begin() + image_size.width * image_size.height * 3, res.end(), 0.0f);")
content = content.replace("std::fill(res.begin(), res.end(), 0.0);", "std::fill(res.begin(), res.end(), 0.0f);")

with open('tools/mtmd/clip.cpp', 'w') as f:
    f.write(content)
