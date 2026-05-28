#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import urllib.request
import platform
import shutil

logger = logging.getLogger("sync_vendor")

HTTPLIB_VERSION = "refs/tags/v0.40.0"

def fallback_python_sync():
    logger.info("Falling back to Python-based sequential sync...")
    vendor = {
        "https://github.com/nlohmann/json/releases/latest/download/json.hpp":     "vendor/nlohmann/json.hpp",
        "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp": "vendor/nlohmann/json_fwd.hpp",

        "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "vendor/stb/stb_image.h",

        "https://github.com/mackron/miniaudio/raw/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d/miniaudio.h": "vendor/miniaudio/miniaudio.h",

        f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/httplib.h": "httplib.h",
        f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/split.py":  "split.py",
        f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/LICENSE":   "vendor/cpp-httplib/LICENSE",

        "https://raw.githubusercontent.com/sheredom/subprocess.h/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h": "vendor/sheredom/subprocess.h",
    }

    for url, filename in vendor.items():
        print(f"downloading {url} to {filename}") # noqa: NP100
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        urllib.request.urlretrieve(url, filename)

    print("Splitting httplib.h...") # noqa: NP100
    try:
        subprocess.check_call([
            sys.executable, "split.py",
            "--extension", "cpp",
            "--out", "vendor/cpp-httplib"
        ])
    except Exception as e:
        print(f"Error: {e}") # noqa: NP100
        sys.exit(1)
    finally:
        if os.path.exists("split.py"):
            os.remove("split.py")
        if os.path.exists("httplib.h"):
            os.remove("httplib.h")

def main():
    llama_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    rusty_dir = os.path.join(llama_path, "rusty")

    bin_name = "sync_vendor.exe" if platform.system() == "Windows" else "sync_vendor"
    rusty_bin = os.path.join(rusty_dir, "target", "release", bin_name)

    if not os.path.exists(rusty_bin):
        if shutil.which("cargo"):
            logger.info("Cargo found. Building the rusty parallel sync_vendor tool for faster execution...")
            try:
                subprocess.run(["cargo", "build", "--release", "--bin", "sync_vendor", "--manifest-path", os.path.join(rusty_dir, "Cargo.toml")], check=True, cwd=rusty_dir)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to build rusty tool: {e}")
                fallback_python_sync()
                return
        else:
            fallback_python_sync()
            return

    try:
        subprocess.run([rusty_bin], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Rusty parallel tool failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
