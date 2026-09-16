#!/usr/bin/env python3
"""
Root entry point for the Astronaut HAR System.

Usage:
    # Run simulated scenarios
    python main.py --mode simulate --scenario correct
    python main.py --mode simulate --scenario skip
    python main.py --mode simulate --scenario wrong

    # Run live webcam / video mode
    python main.py --mode live --source 0
"""
from __future__ import annotations
import os
import sys

# Ensure har_astronaut_system package is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "har_astronaut_system")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from main import run, parse_args

if __name__ == "__main__":
    run(parse_args())
