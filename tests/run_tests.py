#!/usr/bin/env python3
"""
Simple test runner for PPO trading bot
Just runs pytest on all test files
"""

import subprocess
import sys
import os


def main():
    """Run all tests using pytest"""
    print("🧪 PPO TRADING BOT TEST SUITE")
    print("=" * 50)

    # Change to project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    print(f"📂 Running tests from: {project_root}")

    # Run pytest with verbose output
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'
        ], check=False)

        if result.returncode == 0:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n❌ Some tests failed (exit code: {result.returncode})")

        return result.returncode

    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
