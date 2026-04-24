import sys

with open('tools/mtmd/mtmd-audio.cpp', 'r') as f:
    content = f.read()

content = content.replace("std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0);", "std::fill(fft_in.begin() + (n_samples - offset), fft_in.end(), 0.0f);")
content = content.replace("std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0);", "std::fill(samples_padded.begin() + n_samples + stage_2_pad, samples_padded.begin() + n_samples + stage_1_pad + 2 * stage_2_pad, 0.0f);")

with open('tools/mtmd/mtmd-audio.cpp', 'w') as f:
    f.write(content)

with open('tools/mtmd/clip.cpp', 'r') as f:
    content = f.read()

content = content.replace("""                                std::fill(
                                    res_padded.begin() + idx,
                                    res_padded.begin() + idx + (pad_x * 3),
                                    0.0);""", """                                std::fill(
                                    res_padded.begin() + idx,
                                    res_padded.begin() + idx + (pad_x * 3),
                                    0.0f);""")

with open('tools/mtmd/clip.cpp', 'w') as f:
    f.write(content)
