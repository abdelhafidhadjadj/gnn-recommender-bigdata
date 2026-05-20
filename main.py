"""Backward-compatible shim. Canonical entry point is src/main.py."""
import subprocess, sys, os
if __name__ == "__main__":
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "main.py")
    sys.exit(subprocess.run([sys.executable, src] + sys.argv[1:]).returncode)
