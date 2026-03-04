"""
Minimal dev stub for Windows development.
Delegates to the venv Python to run the real server.
This avoids bundling torch/transformers in PyInstaller (fast build, tiny exe).
"""
import subprocess
import sys
import os


def find_project_root():
    if getattr(sys, "frozen", False):
        # Running as PyInstaller bundle from tauri/src-tauri/binaries/
        exe_dir = os.path.dirname(sys.executable)
        return os.path.abspath(os.path.join(exe_dir, "..", "..", ".."))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


if __name__ == "__main__":
    project_root = find_project_root()
    venv_python = os.path.join(project_root, "backend", "venv", "Scripts", "python.exe")

    if not os.path.exists(venv_python):
        print(f"ERROR: venv Python not found at {venv_python}", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [venv_python, "-m", "uvicorn", "backend.main:app"] + sys.argv[1:],
        cwd=project_root,
    )
    sys.exit(result.returncode)
