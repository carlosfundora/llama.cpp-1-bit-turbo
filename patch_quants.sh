sed -i 's/const float32x4_t vd = vdupq_n_f32(d);//g' ggml/src/ggml-quants.c
sed -i 's/const float32x4_t vnd = vdupq_n_f32(-d);//g' ggml/src/ggml-quants.c
sed -i 's/float32x4_t result;/float32x4_t result = vdupq_n_f32(0.0f);/g' ggml/src/ggml-quants.c
sed -i '/const llama_model \* eagle3_target_model = nullptr;/d' src/llama-context.h
