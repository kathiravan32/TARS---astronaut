#!/usr/bin/env python3
"""
Interactive / preset launcher for Astronaut HAR system.
"""
from __future__ import annotations
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "har_astronaut_system")
sys.path.insert(0, PROJECT_DIR)

from main import run, parse_args

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("=" * 60)
        print("🛰 ASTRONAUT HAR - EXPERIMENT MONITORING LAUNCHER")
        print("=" * 60)
        print("Select run mode:")
        print("  1) Simulate Scenario: Correct (Step 1 -> 12)")
        print("  2) Simulate Scenario: Skip    (Skips Step 4)")
        print("  3) Simulate Scenario: Wrong   (Out-of-order sequence)")
        print("  4) Live Mode: Webcam / Video Feed")
        print("  5) Run Automated Test Suite")
        print("=" * 60)
        
        try:
            choice = input("Enter choice [1-5] (default 1): ").strip()
        except Exception:
            choice = "1"
            
        if choice == "2":
            sys.argv.extend(["--mode", "simulate", "--scenario", "skip"])
        elif choice == "3":
            sys.argv.extend(["--mode", "simulate", "--scenario", "wrong"])
        elif choice == "4":
            sys.argv.extend(["--mode", "live", "--source", "0"])
        elif choice == "5":
            import unittest
            from tests.test_system import *
            unittest.main(module="tests.test_system", argv=["run.py"])
            sys.exit(0)
        else:
            sys.argv.extend(["--mode", "simulate", "--scenario", "correct"])

    run(parse_args())
