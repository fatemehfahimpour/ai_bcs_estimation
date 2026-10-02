"""
Environment and Hardware Verification Utility.

Checks and displays runtime environment details, including the operating system,
Python interpreter version and path, PyTorch version, and CUDA/GPU availability.
"""

import platform
import sys
import torch


def check_environment() -> None:
    """
    Inspect and print summary details of the active Python and PyTorch runtime.

    Logs the operating system name and release, Python version, executable path,
    installed PyTorch version, and hardware acceleration device (CUDA or CPU).
    """
    print("-" * 30)
    print("📋 Project Environment Info:")
    print("-" * 30)

    print(f"💻 OS: {platform.system()} {platform.release()}")

    print(f"🐍 Python Version: {sys.version.split()[0]}")
    print(f"📍 Interpreter Path: {sys.executable}")

    print(f"🔥 PyTorch Version: {torch.__version__}")

    device: str = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"⚙️  Running on: {device}")

    print("-" * 30)


if __name__ == "__main__":
    try:
        check_environment()
    except Exception as e:
        print(f"❌ Error: {e}")
