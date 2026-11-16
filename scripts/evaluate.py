"""Portable evaluate entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clingeo.cli import evaluate_main

if __name__ == "__main__":
    evaluate_main()
