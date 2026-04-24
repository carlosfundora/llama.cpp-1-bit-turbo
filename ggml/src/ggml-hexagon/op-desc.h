#ifndef OP_DESC_H
#define OP_DESC_H

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"

#include <string>
#include <stdio.h>

struct op_desc {
    char strides[64 * GGML_MAX_SRC];
    char dims[64 * GGML_MAX_SRC];
    char types[16 * GGML_MAX_SRC];
    char buffs[64 * GGML_MAX_SRC];
    char names[64 * GGML_MAX_SRC];

    int format_tensor_dims(char * str, size_t size, const struct ggml_tensor * t) {
        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, size, "%d:%d", (int) t->ne[0], (int) t->ne[1]);
        } else {
            return snprintf(str, size, "%d:%d:%d:%d", (int) t->ne[0], (int) t->ne[1], (int) t->ne[2], (int) t->ne[3]);
        }
    }

    void format_op_dims(char * str, size_t size, const struct ggml_tensor * t) {
        char * p = str;
        size_t rem = size;
        int n;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            n = format_tensor_dims(p, rem, t->src[0]);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                n = snprintf(p, rem, " x ");
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

                n = format_tensor_dims(p, rem, t->src[i]);
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
            }

            n = snprintf(p, rem, " -> ");
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
        }

        // format self dims separately for better visual alignment
        char self[64];
        format_tensor_dims(self, sizeof(self), t);

        snprintf(p, rem, "%s", self);
    }

    int format_tensor_strides(char * str, size_t size, const struct ggml_tensor * t) {
        const char * c = ggml_is_contiguous(t) ? "" : "!";

        if (t->ne[2] == 1 && t->ne[3] == 1) {
            return snprintf(str, size, "%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], c);
        } else {
            return snprintf(str, size, "%zu:%zu:%zu:%zu%s", (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], c);
        }
    }

    void format_op_strides(char * str, size_t size, const struct ggml_tensor * t) {
        char * p = str;
        size_t rem = size;
        int n;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            n = format_tensor_strides(p, rem, t->src[0]);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                n = snprintf(p, rem, " x ");
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

                n = format_tensor_strides(p, rem, t->src[i]);
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
            }

            n = snprintf(p, rem, " -> ");
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
        }

        // format self dims separately for better visual alignment
        char self[64];
        format_tensor_strides(self, sizeof(self), t);

        snprintf(p, rem, "%s", self);
    }

    void format_op_types(char * str, size_t size, const struct ggml_tensor * t) {
        char * p = str;
        size_t rem = size;
        int n;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            n = snprintf(p, rem, "%s", ggml_type_name(t->src[0]->type));
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                n = snprintf(p, rem, " x ");
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

                n = snprintf(p, rem, "%s", ggml_type_name(t->src[i]->type));
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
            }

            n = snprintf(p, rem, " -> ");
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
        }

        snprintf(p, rem, "%s", ggml_type_name(t->type));
    }

    const char * tensor_buff_name(const struct ggml_tensor * t) {
        if (t->buffer) {
            return ggml_backend_buffer_name(t->buffer);
        }
        return "NONE";
    }

    void format_op_buffs(char * str, size_t size, const struct ggml_tensor * t) {
        char * p = str;
        size_t rem = size;
        int n;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            n = snprintf(p, rem, "%s", tensor_buff_name(t->src[0]));
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                n = snprintf(p, rem, " x ");
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

                n = snprintf(p, rem, "%s", tensor_buff_name(t->src[i]));
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
            }

            n = snprintf(p, rem, " -> ");
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
        }

        snprintf(p, rem, "%s", tensor_buff_name(t));
    }

    void format_op_names(char * str, size_t size, const struct ggml_tensor * t) {
        char * p = str;
        size_t rem = size;
        int n;

        // append src0 and src1 (if any)
        if (t->src[0]) {
            n = snprintf(p, rem, "%s", t->src[0]->name);
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

            for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
                n = snprintf(p, rem, " x ");
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }

                n = snprintf(p, rem, "%s", t->src[i]->name);
                if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
            }

            n = snprintf(p, rem, " -> ");
            if (n > 0 && (size_t)n < rem) { p += n; rem -= n; } else { return; }
        }

        snprintf(p, rem, "%s", t->name);
    }

    void format(const ggml_tensor * op) {
        format_op_dims(dims, sizeof(dims), op);
        format_op_strides(strides, sizeof(strides), op);
        format_op_types(types, sizeof(types), op);
        format_op_buffs(buffs, sizeof(buffs), op);
        format_op_names(names, sizeof(names), op);
    }

    op_desc() {}
    op_desc(const ggml_tensor * op) { format(op); }
};

#endif // OP_DESC_H
