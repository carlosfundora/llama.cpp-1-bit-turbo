#!/usr/bin/env python3
import sys
import os
import subprocess
import logging

def main(args):
    llama_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    rusty_dir = os.path.join(llama_path, "rusty")
    import platform
    bin_name = "gen_unicode_data.exe" if platform.system() == "Windows" else "gen_unicode_data"
    rusty_bin = os.path.join(rusty_dir, "target", "release", bin_name)

    if not os.path.exists(rusty_bin):
        logging.info("Rust binary not found. Compiling gen_unicode_data...")
        subprocess.run(["cargo", "build", "--release", "--bin", "gen_unicode_data", "--manifest-path", os.path.join(rusty_dir, "Cargo.toml")], check=True, cwd=rusty_dir)

    cmd = [rusty_bin] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
