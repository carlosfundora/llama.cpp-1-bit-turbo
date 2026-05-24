#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install ty
ty check --output-format=github
