Oh!
Wait, `can_use_vector_kernel` is `Q->ne[0] <= 256 && Q->ne[0] % 64 == 0`.
So `Q->ne[0]` can only be 64, 128, 192, 256.
BUT `FATTN_VEC_CASES_ALL_D` only has `64`, `128`, `256`!
It is MISSING `192`!
If `Q->ne[0] == 192`, `can_use_vector_kernel` is TRUE, and it returns `BEST_FATTN_KERNEL_VEC`.
Then `ggml_cuda_flash_attn_ext_vec` gets called.
And it goes through the macros, doesn't match `192`, and reaches:
`GGML_ABORT("fatal error");`
Is this true??? Let's verify `192`.
Does any model use head dimension 192?
Maybe some models do, but wait! Does `ggml_cuda_get_best_fattn_kernel` abort for 192?
Let's see `ggml_cuda_get_best_fattn_kernel` switch:
```cpp
    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
```
Wait, `192` is not in this switch!
If `K->ne[0] == 192`, it hits `default: return BEST_FATTN_KERNEL_NONE;`
So `Q->ne[0] == 192` is NOT supported at all! It will just fall back to standard attention.
So there is no bug about `192` because the switch filters it out.

Then my fix for RDNA2 using `BEST_FATTN_KERNEL_VEC` is perfectly safe!

Let's do these three things:
1. Fix `warp_size` in `ggml-cuda.cu` (line 2373)
2. Disable CUDA graph capture for AMD in `ggml-cuda.cu`
3. Improve RDNA2 decode performance by returning `BEST_FATTN_KERNEL_VEC` for small batch sizes in `ggml_cuda_get_best_fattn_kernel` in `fattn.cu`.

Let's prepare the patches and review the plan!
