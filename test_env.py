import sys
import torch
import platform

def check_environment():
    print("-" * 30)
    print("📋 Project Environment Info:")
    print("-" * 30)
    
    print(f"💻 OS: {platform.system()} {platform.release()}")
    
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    print(f"📍 Interpreter Path: {sys.executable}")
    
    print(f"🔥 PyTorch Version: {torch.__version__}")
    
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    print(f"⚙️  Running on: {device}")
    
    print("-" * 30)

if __name__ == "__main__":
    try:
        check_environment()
    except Exception as e:
        print(f"❌ Error: {e}")
