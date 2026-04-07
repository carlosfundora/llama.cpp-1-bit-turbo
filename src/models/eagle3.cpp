#include "models.h"

// EAGLE3 Encoder: fc projection only
// Input: target_features [fc_input_size, n_tokens] (concatenated hidden states from target layers)
// Output: g_embeddings [hidden, n_tokens]
llm_build_eagle3_encode::llm_build_eagle3_encode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    // Compute fc_input_size from extract_layers count × hidden_size
    int64_t n_extract = 0;
    for (int i = 0; i < 3; i++) {
        if (hparams.eagle3_extract_layers[i] >= 0) n_extract++;
    }
    if (n_extract == 0) n_extract = 3; // fallback
    const int64_t fc_input_size = n_extract * n_embd;

    // Create custom input tensor for target features [fc_input_size, n_tokens]
    // This is passed as batch.embd by the speculative loop
    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, fc_input_size, n_tokens);
    ggml_set_input(inp);
    ggml_set_name(inp, "inp_eagle3_features");

    // fc projection: [fc_input_size, n_tokens] -> [hidden, n_tokens]
    // fc.weight shape is [hidden, fc_input_size] so mul_mat gives [hidden, n_tokens]
    ggml_tensor * cur = ggml_mul_mat(ctx0, model.fc, inp);
    cb(cur, "eagle3_fc_out", -1);

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}

// EAGLE3 Decoder: 1-layer transformer with 2×hidden input dimension
// The decoder concatenates token_embd(token) + g_embeddings as input to attention
// Attention Q/K/V projections have input dim = 2*hidden
// Residual is on g_embeddings (not the concatenated input)
llm_build_eagle3_decode::llm_build_eagle3_decode(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;

    // Token embedding for the draft token
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    // inp_pos for RoPE
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // g_embeddings: from encoder fc output (first step) or previous decoder prenorm (autoregressive).
    // This is a proper graph input — filled by the speculative loop via set_input before each decode.
    ggml_tensor * g_embd = build_inp_eagle3_g_embd();

    for (int il = 0; il < n_layer; ++il) {
        // 1. Normalize token embedding with attn_norm (input_layernorm)
        ggml_tensor * embd_norm = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(embd_norm, "embd_norm", il);

        // 2. Normalize g_embeddings with hidden_norm
        ggml_tensor * g_embd_norm = build_norm(g_embd,
                model.layers[il].eagle3_hidden_norm, NULL,
                LLM_NORM_RMS, il);
        cb(g_embd_norm, "g_embd_norm", il);

        // 3. Concatenate: [embd_norm; g_embd_norm] → [2*n_embd, n_tokens]
        ggml_tensor * attn_input = ggml_concat(ctx0, embd_norm, g_embd_norm, 0);
        cb(attn_input, "attn_input_concat", il);

        // 4. Self-attention with 2×hidden input
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, attn_input);
            cb(Qcur, "Qcur", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, attn_input);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, attn_input);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f/sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            g_embd = ggml_get_rows(ctx0, g_embd, inp_out_ids);
        }

        // 5. Residual on g_embeddings (NOT on token embedding or concat)
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, g_embd);
        cb(ffn_inp, "ffn_inp", il);

        // FFN
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // For autoregressive: prenorm output becomes next g_embeddings
        g_embd = cur;
        inpL = cur;
    }

    cur = inpL;

    // Store PRENORM output for autoregressive g_embeddings recurrence.
    // The speculative loop retrieves this via llama_get_embeddings_ith()
    // and feeds it back as g_embeddings for the next decode step.
    // Must be prenorm — the decoder applies hidden_norm to g_embd input.
    // (Matches upstream PR-18039: t_embd = prenorm, BEFORE output_norm)
    ggml_set_output(cur);
    res->t_embd = cur;

    // Output norm + lm_head → draft logits (post-norm, for sampling only)
    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
