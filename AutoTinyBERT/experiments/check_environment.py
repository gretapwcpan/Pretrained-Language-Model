#!/usr/bin/env python3
"""
Environment validation script for AutoTinyBERT
Run this BEFORE installing packages to check what you already have
"""

import sys
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 7:
        print("✅ Python version OK (3.7+ required)")
        return True
    else:
        print("❌ Python 3.7+ required")
        return False

def check_cuda():
    """Check CUDA availability and version"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\n✅ NVIDIA GPU detected")
            # Try to get CUDA version
            for line in result.stdout.split('\n'):
                if 'CUDA Version' in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("\n⚠️ No NVIDIA GPU detected (CPU mode will be used)")
            return False
    except FileNotFoundError:
        print("\n⚠️ nvidia-smi not found (CPU mode will be used)")
        return False

def check_package(package_name):
    """Check if a package is installed and get version"""
    try:
        if package_name == 'torch':
            import torch
            print(f"✅ {package_name}: {torch.__version__}")
            # Check CUDA support in PyTorch
            if torch.cuda.is_available():
                print(f"   └─ CUDA support: Yes (CUDA {torch.version.cuda})")
            else:
                print(f"   └─ CUDA support: No")
            return True
        elif package_name == 'transformers':
            import transformers
            print(f"✅ {package_name}: {transformers.__version__}")
            return True
        elif package_name == 'numpy':
            import numpy
            print(f"✅ {package_name}: {numpy.__version__}")
            return True
        elif package_name == 'tqdm':
            import tqdm
            print(f"✅ {package_name}: {tqdm.__version__}")
            return True
        elif package_name == 'tensorboard':
            import tensorboard
            print(f"✅ {package_name}: {tensorboard.__version__}")
            return True
        elif package_name == 'tensorboardX':
            import tensorboardX
            print(f"✅ {package_name}: {tensorboardX.__version__}")
            return True
        elif package_name == 'apex':
            import apex
            print(f"✅ {package_name}: installed")
            return True
        else:
            __import__(package_name)
            print(f"✅ {package_name}: installed")
            return True
    except ImportError:
        print(f"❌ {package_name}: NOT installed")
        return False

def main():
    print("=" * 60)
    print("AutoTinyBERT Environment Validation")
    print("=" * 60)
    
    # Check Python version
    print("\n📌 Python Environment:")
    python_ok = check_python_version()
    
    # Check CUDA/GPU
    print("\n📌 GPU/CUDA Status:")
    has_gpu = check_cuda()
    
    # Check installed packages
    print("\n📌 Package Status:")
    required_packages = ['torch', 'transformers', 'numpy', 'tqdm', 'tensorboard']
    optional_packages = ['tensorboardX', 'apex']
    
    print("\nRequired packages:")
    missing_required = []
    for pkg in required_packages:
        if not check_package(pkg):
            missing_required.append(pkg)
    
    print("\nOptional packages:")
    missing_optional = []
    for pkg in optional_packages:
        if not check_package(pkg):
            missing_optional.append(pkg)
    
    # Provide installation recommendations
    print("\n" + "=" * 60)
    print("📋 Installation Recommendations:")
    print("=" * 60)
    
    if not missing_required:
        print("\n✅ All required packages are installed!")
    else:
        print(f"\n⚠️ Missing required packages: {', '.join(missing_required)}")
        print("\nInstall with:")
        
        # Check if torch needs special installation
        if 'torch' in missing_required:
            if has_gpu:
                print("# For GPU (check your CUDA version first):")
                print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            else:
                print("# For CPU:")
                print("pip install torch torchvision torchaudio")
            
            # Remove torch from the list for the next command
            other_packages = [p for p in missing_required if p != 'torch']
            if other_packages:
                print(f"\n# Then install other packages:")
                print(f"pip install {' '.join(other_packages)}")
        else:
            print(f"pip install {' '.join(missing_required)}")
    
    if missing_optional:
        print(f"\n⚠️ Missing optional packages: {', '.join(missing_optional)}")
        if 'apex' in missing_optional and has_gpu:
            print("\nFor apex (GPU acceleration):")
            print("# Note: Install AFTER PyTorch is installed")
            print("git clone https://github.com/NVIDIA/apex")
            print("cd apex")
            print("pip install -v --no-cache-dir --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext\" ./")
        if 'tensorboardX' in missing_optional:
            print("\nFor tensorboardX:")
            print("pip install tensorboardX")
    
    # Check if PyTorch CUDA matches system CUDA
    try:
        import torch
        if torch.cuda.is_available() and has_gpu:
            print("\n✅ PyTorch CUDA is properly configured")
        elif not torch.cuda.is_available() and has_gpu:
            print("\n⚠️ GPU detected but PyTorch doesn't have CUDA support")
            print("   Consider reinstalling PyTorch with CUDA support")
    except ImportError:
        pass
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
