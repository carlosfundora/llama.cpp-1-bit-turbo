import subprocess
import sys

try:
    subprocess.run(["python", "-m", "pip", "uninstall", "-y", "flake8", "flake8-no-print", "importlib-metadata"])
    subprocess.run(["python", "-m", "pip", "install", "flake8>=5.0.0"])
    subprocess.run(["python", "-m", "pip", "install", "flake8-no-print>=0.1.2"])
except Exception as e:
    print(e)
