//! Build script for `rs_llama_cpp_sys`.
//!
//! The working tree IS the llama.cpp 1-bit-turbo fork and the HIP artifacts are ALREADY built at
//! `build-hip/bin/*.so`. So we do NOT download a submodule or run cmake (the llama-cpp-rs pattern):
//! we bindgen the in-tree headers and link the prebuilt shared libraries directly.

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest
        .parent()
        .expect("rs_llama_cpp_sys must live under the fork root")
        .to_path_buf();

    let inc = root.join("include");
    let ggml_inc = root.join("ggml").join("include");
    let lib_bin = root.join("build-hip").join("bin");

    assert!(
        inc.join("llama.h").exists(),
        "in-tree header {} not found",
        inc.join("llama.h").display()
    );
    assert!(
        lib_bin.join("libllama.so").exists(),
        "prebuilt {} not found — build the HIP tree first",
        lib_bin.join("libllama.so").display()
    );

    // --- bindgen: the C API surface (llama_* / ggml_* / gguf_*) ---
    let bindings = bindgen::Builder::default()
        .header(manifest.join("wrapper.h").to_string_lossy())
        .clang_arg(format!("-I{}", inc.display()))
        .clang_arg(format!("-I{}", ggml_inc.display()))
        .clang_arg("-x")
        .clang_arg("c")
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_function("gguf_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("gguf_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .prepend_enum_name(false)
        .derive_default(true)
        .generate_comments(false)
        .generate()
        .expect("bindgen failed to generate llama.cpp bindings");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    // --- link the prebuilt HIP artifacts ---
    println!("cargo:rustc-link-search=native={}", lib_bin.display());
    for lib in ["llama", "ggml", "ggml-base", "ggml-cpu", "ggml-hip", "mtmd"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    // ggml/llama are C++ TUs.
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // Resolve the .so at runtime without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_bin.display());

    println!("cargo:rerun-if-changed={}", manifest.join("wrapper.h").display());
    println!("cargo:rerun-if-changed={}", inc.join("llama.h").display());
    println!("cargo:rerun-if-changed={}", inc.join("llama-cpp.h").display());
}
