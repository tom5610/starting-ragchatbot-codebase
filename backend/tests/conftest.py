import sys
import os

# Make backend/ importable from any test file without package install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
