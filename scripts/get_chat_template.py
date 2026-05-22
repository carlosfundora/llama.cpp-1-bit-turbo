#!/usr/bin/env python3
'''
  Fetches the Jinja chat template of a HuggingFace model.
  If a model has multiple chat templates, you can specify the variant name.

  Syntax:
    ./scripts/get_chat_template.py model_id [variant]

  Examples:
    ./scripts/get_chat_template.py CohereForAI/c4ai-command-r-plus tool_use
    ./scripts/get_chat_template.py microsoft/Phi-3.5-mini-instruct
'''

import sys
import os
import subprocess
import logging

def main(args):
    if len(args) < 1:
        raise ValueError("Please provide a model ID and an optional variant name")

    llama_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    rusty_dir = os.path.join(llama_path, "rusty")
    import platform
    bin_name = "get_chat_template.exe" if platform.system() == "Windows" else "get_chat_template"
    rusty_bin = os.path.join(rusty_dir, "target", "release", bin_name)

    if not os.path.exists(rusty_bin):
        logging.info("Rust binary not found. Compiling get_chat_template...")
        subprocess.run(["cargo", "build", "--release", "--bin", "get_chat_template", "--manifest-path", os.path.join(rusty_dir, "Cargo.toml")], check=True, cwd=rusty_dir)

    cmd = [rusty_bin] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == '__main__':
    main(sys.argv[1:])
