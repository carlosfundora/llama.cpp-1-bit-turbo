#!/bin/bash

# Fix implicit conversion warnings to int in ggml-blas
sed -i 's/__builtin_popcountll/__popcnt64/g' common/phantom.h
