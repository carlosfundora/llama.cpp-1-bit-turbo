//! Build script for `rs_embedding_server`.
//!
//! `rs_llama_cpp_sys` emits the link-search + link-lib lines (transitive), but its rpath
//! (`cargo:rustc-link-arg=-Wl,-rpath,...`) is NOT transitive to a dependent binary. So this binary
//! would fail at runtime with `libllama.so.0: cannot open shared object file` unless launched with
//! `LD_LIBRARY_PATH=<repo>/build-hip/bin`. Re-emit the rpath here so the produced binary is
//! self-contained (resolves the HIP .so at runtime on its own).

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    // ../  == the llama.cpp fork root that holds build-hip/bin/*.so
    let lib_bin = manifest
        .parent()
        .expect("rs_embedding_server must live under the fork root")
        .join("build-hip")
        .join("bin");
    if lib_bin.join("libllama.so").exists() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_bin.display());
    } else {
        println!(
            "cargo:warning=rs_embedding_server: {} not found; binary will need LD_LIBRARY_PATH at runtime",
            lib_bin.display()
        );
    }
}
