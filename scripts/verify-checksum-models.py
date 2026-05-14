#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import hashlib
import platform
import shutil

logger = logging.getLogger("verify-checksum-models")

def fallback_sha256sum(file_path):
    block_size = 16 * 1024 * 1024  # 16 MB block size
    b = bytearray(block_size)
    file_hash = hashlib.sha256()
    mv = memoryview(b)
    with open(file_path, 'rb', buffering=0) as f:
        while True:
            n = f.readinto(mv)
            if not n:
                break
            file_hash.update(mv[:n])
    return file_hash.hexdigest()

def fallback_python_verify(llama_path):
    logger.info("Falling back to Python-based sequential verification...")
    hash_list_file = os.path.join(llama_path, "SHA256SUMS")
    if not os.path.exists(hash_list_file):
        logger.error(f"Hash list file not found: {hash_list_file}")
        sys.exit(1)

    with open(hash_list_file, "r") as f:
        hash_list = f.read().splitlines()

    results = []
    for line in hash_list:
        if not line.strip() or "  " not in line:
            continue
        hash_value, filename = line.split("  ")
        file_path = os.path.join(llama_path, filename)

        logger.info(f"Verifying the checksum of {file_path}")

        if os.path.exists(file_path):
            file_hash = fallback_sha256sum(file_path)
            if file_hash == hash_value:
                valid_checksum = "V"
                file_missing = ""
            else:
                valid_checksum = ""
                file_missing = ""
        else:
            valid_checksum = ""
            file_missing = "X"

        results.append({
            "filename": filename,
            "valid checksum": valid_checksum,
            "file missing": file_missing
        })

    print("filename".ljust(40) + "valid checksum".center(20) + "file missing".center(20)) # noqa: NP100
    print("-" * 80) # noqa: NP100
    for r in results:
        print(f"{r['filename']:40} {r['valid checksum']:^20} {r['file missing']:^20}") # noqa: NP100

def main():
    llama_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    rusty_dir = os.path.join(llama_path, "rusty")

    bin_name = "rusty.exe" if platform.system() == "Windows" else "rusty"
    rusty_bin = os.path.join(rusty_dir, "target", "release", bin_name)

    # Try using compiled binary if it exists
    if not os.path.exists(rusty_bin):
        # Try compiling if cargo is installed
        if shutil.which("cargo"):
            logger.info("Cargo found. Building the rusty parallel checksum tool for faster execution...")
            try:
                subprocess.run(["cargo", "build", "--release", "--manifest-path", os.path.join(rusty_dir, "Cargo.toml")], check=True, cwd=rusty_dir)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build rusty tool: {e}")
                return fallback_python_verify(llama_path)
        else:
            return fallback_python_verify(llama_path)

    # Run the rusty tool
    try:
        subprocess.run([rusty_bin, "--base-dir", llama_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Rusty parallel tool failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
