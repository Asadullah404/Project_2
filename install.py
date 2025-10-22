
# #!/usr/bin/env python3
# """
# install_dependencies.py - Comprehensive Dependency Installer for Neural Audio Codec
# Optimized for Python 3.10.11 and NVIDIA GTX 1060 6GB

# This script will:
# 1. Verify Python version
# 2. Detect GPU and CUDA capabilities
# 3. Install PyTorch with appropriate CUDA support
# 4. Install all required packages
# 5. Verify installations
# 6. Provide detailed troubleshooting info if anything fails

# Usage:
#     python install_dependencies.py              # Auto-detect and install
#     python install_dependencies.py --cpu-only   # Force CPU-only PyTorch
#     python install_dependencies.py --cuda-11    # Force CUDA 11.8
#     python install_dependencies.py --cuda-12    # Force CUDA 12.1
# """

# import sys
# import subprocess
# import platform
# import os
# import argparse
# import re
# from pathlib import Path

# # ANSI color codes for pretty output
# class Colors:
#     HEADER = '\033[95m'
#     OKBLUE = '\033[94m'
#     OKCYAN = '\033[96m'
#     OKGREEN = '\033[92m'
#     WARNING = '\033[93m'
#     FAIL = '\033[91m'
#     ENDC = '\033[0m'
#     BOLD = '\033[1m'
#     UNDERLINE = '\033[4m'

# def print_header(text):
#     """Print a formatted header."""
#     print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
#     print(f"{text}")
#     print(f"{'=' * 80}{Colors.ENDC}\n")

# def print_success(text):
#     """Print success message."""
#     print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

# def print_warning(text):
#     """Print warning message."""
#     print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")

# def print_error(text):
#     """Print error message."""
#     print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")

# def print_info(text):
#     """Print info message."""
#     print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")

# def run_command(cmd, timeout=300, capture_output=True):
#     """
#     Run a shell command and return result.
    
#     Args:
#         cmd: Command as list of strings
#         timeout: Timeout in seconds
#         capture_output: Whether to capture output
    
#     Returns:
#         Tuple of (success: bool, stdout: str, stderr: str)
#     """
#     try:
#         result = subprocess.run(
#             cmd,
#             capture_output=capture_output,
#             text=True,
#             timeout=timeout,
#             check=False,
#             encoding='utf-8',
#             errors='replace'
#         )
#         return (result.returncode == 0, result.stdout, result.stderr)
#     except subprocess.TimeoutExpired:
#         return (False, "", "Command timed out")
#     except Exception as e:
#         return (False, "", str(e))

# def check_python_version():
#     """Check if Python version is compatible."""
#     print_header("CHECKING PYTHON VERSION")
    
#     version = sys.version_info
#     version_str = f"{version.major}.{version.minor}.{version.micro}"
    
#     print(f"Detected Python version: {version_str}")
#     print(f"Running on: {platform.system()} {platform.machine()}")
#     print(f"Python executable: {sys.executable}")
    
#     if version.major != 3:
#         print_error(f"Python 3.x required, but found Python {version.major}")
#         return False
    
#     if version.minor < 8:
#         print_error(f"Python 3.8+ required, but found Python {version_str}")
#         return False
    
#     if version.minor > 11:
#         print_warning(f"Python {version_str} detected (script optimized for 3.10.11)")
#         print_warning("Some packages may have compatibility issues")
    
#     if version.minor == 10 and version.micro == 11:
#         print_success(f"Perfect! Python {version_str} is the target version")
#     else:
#         print_warning(f"Python {version_str} detected (recommended: 3.10.11)")
#         print_info("Should work, but some differences may occur")
    
#     return True

# def check_pip():
#     """Ensure pip is installed and up to date."""
#     print_header("CHECKING PIP")
    
#     # Check if pip is available
#     success, stdout, stderr = run_command([sys.executable, "-m", "pip", "--version"])
    
#     if not success:
#         print_error("pip is not available")
#         print_info("Installing pip...")
        
#         # Try to install pip
#         success, _, _ = run_command([sys.executable, "-m", "ensurepip", "--default-pip"])
#         if not success:
#             print_error("Failed to install pip")
#             return False
    
#     print_success("pip is available")
#     print(f"  {stdout.strip()}")
    
#     # Upgrade pip
#     print_info("Upgrading pip to latest version...")
#     success, _, _ = run_command([
#         sys.executable, "-m", "pip", "install", "--upgrade", "pip"
#     ])
    
#     if success:
#         print_success("pip upgraded successfully")
#     else:
#         print_warning("Failed to upgrade pip, but continuing...")
    
#     return True

# def detect_nvidia_gpu():
#     """
#     Detect NVIDIA GPU and CUDA version.
    
#     Returns:
#         Dict with keys: has_nvidia, gpu_name, cuda_version, compute_capability
#     """
#     print_header("DETECTING NVIDIA GPU")
    
#     result = {
#         'has_nvidia': False,
#         'gpu_name': None,
#         'cuda_version': None,
#         'compute_capability': None,
#         'driver_version': None
#     }
    
#     # Try nvidia-smi
#     success, stdout, stderr = run_command(["nvidia-smi"], timeout=10)
    
#     if not success:
#         print_warning("nvidia-smi not found or failed")
#         print_info("No NVIDIA GPU detected or drivers not installed")
#         return result
    
#     print_success("nvidia-smi found")
    
#     # Parse nvidia-smi output
#     lines = stdout.split('\n')
    
#     # Extract driver version
#     for line in lines:
#         if 'Driver Version:' in line:
#             match = re.search(r'Driver Version:\s+(\d+\.\d+)', line)
#             if match:
#                 result['driver_version'] = match.group(1)
#                 print(f"  Driver Version: {result['driver_version']}")
        
#         if 'CUDA Version:' in line:
#             match = re.search(r'CUDA Version:\s+(\d+\.\d+)', line)
#             if match:
#                 result['cuda_version'] = match.group(1)
#                 print(f"  CUDA Version: {result['cuda_version']}")
    
#     # Get GPU name
#     success, stdout, stderr = run_command([
#         "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"
#     ], timeout=10)
    
#     if success:
#         lines = stdout.strip().split('\n')
#         if lines:
#             parts = lines[0].split(',')
#             if len(parts) >= 1:
#                 result['gpu_name'] = parts[0].strip()
#                 print(f"  GPU Name: {result['gpu_name']}")
#             if len(parts) >= 2:
#                 result['compute_capability'] = parts[1].strip()
#                 print(f"  Compute Capability: {result['compute_capability']}")
        
#         result['has_nvidia'] = True
        
#         # Check if it's GTX 1060
#         if 'GTX 1060' in result['gpu_name'] or '1060' in result['gpu_name']:
#             print_success("GTX 1060 detected - perfect for this project!")
#             print_info("VRAM: 6GB (make sure you have the 6GB version, not 3GB)")
#         elif result['compute_capability']:
#             cc_major = float(result['compute_capability'].split('.')[0]) if '.' in result['compute_capability'] else 0
#             if cc_major >= 6.0:
#                 print_success(f"GPU detected with compute capability {result['compute_capability']}")
#                 print_info("This GPU should work well with the neural codec")
#             else:
#                 print_warning(f"GPU compute capability {result['compute_capability']} may be too old")
#                 print_warning("CUDA 11.8+ requires compute capability 3.5+")
#     else:
#         print_warning("Could not query GPU details")
    
#     return result

# def determine_cuda_version(gpu_info, force_cpu=False, force_cuda=None):
#     """
#     Determine which CUDA version to use for PyTorch.
    
#     Args:
#         gpu_info: GPU info dict from detect_nvidia_gpu()
#         force_cpu: Force CPU-only installation
#         force_cuda: Force specific CUDA version ('11', '12', etc.)
    
#     Returns:
#         String: 'cpu', 'cu118', 'cu121', etc.
#     """
#     if force_cpu:
#         print_info("CPU-only mode forced by user")
#         return 'cpu'
    
#     if force_cuda:
#         cuda_map = {
#             '11': 'cu118',
#             '11.8': 'cu118',
#             '118': 'cu118',
#             '12': 'cu121',
#             '12.1': 'cu121',
#             '121': 'cu121',
#         }
#         cuda_version = cuda_map.get(force_cuda, force_cuda)
#         print_info(f"CUDA version forced by user: {cuda_version}")
#         return cuda_version
    
#     if not gpu_info['has_nvidia']:
#         print_info("No NVIDIA GPU detected, using CPU version")
#         return 'cpu'
    
#     # GTX 1060 has compute capability 6.1
#     # Best compatibility with CUDA 11.8
#     if gpu_info['compute_capability']:
#         cc = float(gpu_info['compute_capability'])
#         if cc >= 6.0:
#             print_info("Using CUDA 11.8 (best compatibility for GTX 1060)")
#             return 'cu118'
#         else:
#             print_warning(f"Compute capability {cc} < 6.0")
#             print_warning("CUDA 11.8 requires compute capability 3.5+")
#             return 'cpu'
    
#     # Default to CUDA 11.8 if we detected NVIDIA but couldn't get compute cap
#     print_info("Using CUDA 11.8 (default for detected NVIDIA GPU)")
#     return 'cu118'

# def install_pytorch(cuda_version='cu118'):
#     """
#     Install PyTorch with appropriate CUDA support.
    
#     Args:
#         cuda_version: 'cpu', 'cu118', 'cu121', etc.
#     """
#     print_header(f"INSTALLING PYTORCH (CUDA version: {cuda_version})")
    
#     # PyTorch version compatible with Python 3.10.11
#     pytorch_version = "2.0.1"  # Stable version with good Python 3.10 support
#     torchaudio_version = "2.0.2"
    
#     if cuda_version == 'cpu':
#         print_info(f"Installing PyTorch {pytorch_version} (CPU-only)...")
#         cmd = [
#             sys.executable, "-m", "pip", "install",
#             f"torch=={pytorch_version}",
#             f"torchaudio=={torchaudio_version}",
#             "--index-url", "https://download.pytorch.org/whl/cpu"
#         ]
#     elif cuda_version == 'cu118':
#         print_info(f"Installing PyTorch {pytorch_version} with CUDA 11.8...")
#         cmd = [
#             sys.executable, "-m", "pip", "install",
#             f"torch=={pytorch_version}",
#             f"torchaudio=={torchaudio_version}",
#             "--index-url", "https://download.pytorch.org/whl/cu118"
#         ]
#     elif cuda_version == 'cu121':
#         print_info(f"Installing PyTorch {pytorch_version} with CUDA 12.1...")
#         cmd = [
#             sys.executable, "-m", "pip", "install",
#             f"torch=={pytorch_version}",
#             f"torchaudio=={torchaudio_version}",
#             "--index-url", "https://download.pytorch.org/whl/cu121"
#         ]
#     else:
#         print_error(f"Unknown CUDA version: {cuda_version}")
#         return False
    
#     print_info("This may take several minutes...")
#     print_info("Command: " + " ".join(cmd))
    
#     success, stdout, stderr = run_command(cmd, timeout=600)
    
#     if success:
#         print_success("PyTorch installed successfully")
        
#         # Verify installation with better error handling
#         print_info("Verifying PyTorch installation...")
        
#         # Try importing in the same Python process first
#         try:
#             import torch
#             print_info("Verification (direct import):")
#             print(f"    PyTorch {torch.__version__}")
#             print(f"    CUDA available: {torch.cuda.is_available()}")
#             if torch.cuda.is_available():
#                 print(f"    CUDA version: {torch.version.cuda}")
#                 print(f"    Device: {torch.cuda.get_device_name(0)}")
#             else:
#                 print(f"    Device: CPU")
#             return True
#         except ImportError as e:
#             print_warning(f"Direct import failed: {e}")
#             print_info("This might be due to Windows Store Python path issues")
#             print_info("PyTorch is installed, but you may need to restart your terminal")
            
#             # Still try subprocess verification
#             verify_cmd = [
#                 sys.executable, "-c",
#                 "import torch; "
#                 "print(f'PyTorch: {torch.__version__}'); "
#                 "print(f'CUDA: {torch.cuda.is_available()}'); "
#                 "print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
#             ]
            
#             success, stdout, stderr = run_command(verify_cmd, timeout=30)
#             if success:
#                 print_info("Verification (subprocess):")
#                 for line in stdout.strip().split('\n'):
#                     print(f"    {line}")
#                 return True
#             else:
#                 print_warning("Subprocess verification also failed")
#                 print_warning("PyTorch is installed but may need terminal restart")
#                 return True  # Still return True since install succeeded
#     else:
#         print_error("PyTorch installation failed")
#         print_error(f"Error: {stderr[:500]}")
        
#         print_info("\nTroubleshooting:")
#         print_info("1. Try manual installation:")
#         print_info(f"   {' '.join(cmd)}")
#         print_info("2. Check PyTorch website: https://pytorch.org/get-started/locally/")
#         print_info("3. Try CPU version: python install_dependencies.py --cpu-only")
        
#         return False

# def install_package(package_spec, package_name=None, timeout=120):
#     """
#     Install a Python package.
    
#     Args:
#         package_spec: Package specification (e.g., "numpy>=1.21.0")
#         package_name: Display name (defaults to package_spec)
#         timeout: Installation timeout in seconds
    
#     Returns:
#         bool: True if successful
#     """
#     if package_name is None:
#         package_name = package_spec.split('[')[0].split('=')[0].split('>')[0].split('<')[0]
    
#     print(f"\n  Installing {package_name}...", end=" ", flush=True)
    
#     cmd = [sys.executable, "-m", "pip", "install", package_spec]
#     success, stdout, stderr = run_command(cmd, timeout=timeout)
    
#     if success:
#         print_success("OK")
#         return True
#     else:
#         print_error("FAILED")
#         # Only show first 200 chars of error
#         if stderr:
#             print_error(f"    Error: {stderr[:200]}")
#         return False

# def install_core_packages():
#     """Install core required packages."""
#     print_header("INSTALLING CORE PACKAGES")
    
#     # Core numerical and scientific packages
#     packages = [
#         ("numpy>=1.21.0,<2.0.0", "NumPy"),
#         ("scipy>=1.7.0,<1.12.0", "SciPy"),
#         ("matplotlib>=3.3.0,<3.8.0", "Matplotlib"),
#     ]
    
#     results = {}
#     for spec, name in packages:
#         results[name] = install_package(spec, name)
    
#     return all(results.values())

# def install_audio_packages():
#     """Install audio processing packages."""
#     print_header("INSTALLING AUDIO PACKAGES")
    
#     packages = [
#         ("sounddevice>=0.4.0", "sounddevice"),
#         ("soundfile>=0.10.0", "soundfile"),
#         ("librosa>=0.9.0,<0.11.0", "librosa"),
#     ]
    
#     results = {}
#     for spec, name in packages:
#         results[name] = install_package(spec, name, timeout=180)
    
#     return all(results.values())

# def install_utility_packages():
#     """Install utility packages."""
#     print_header("INSTALLING UTILITY PACKAGES")
    
#     packages = [
#         ("tqdm>=4.60.0", "tqdm"),
#         ("Pillow>=8.0.0", "Pillow"),
#     ]
    
#     results = {}
#     for spec, name in packages:
#         results[name] = install_package(spec, name)
    
#     return all(results.values())

# def install_network_packages():
#     """Install networking packages."""
#     print_header("INSTALLING NETWORK PACKAGES")
    
#     packages = [
#         ("aiohttp>=3.8.0,<4.0.0", "aiohttp"),
#     ]
    
#     results = {}
#     for spec, name in packages:
#         results[name] = install_package(spec, name)
    
#     # Optional packages (don't fail if these don't install)
#     optional_packages = [
#         ("aiortc>=1.3.0", "aiortc"),
#     ]
    
#     for spec, name in optional_packages:
#         print_info(f"Attempting to install optional package: {name}")
#         install_package(spec, name)  # Don't track result
    
#     return all(results.values())

# def install_metrics_packages():
#     """Install audio metrics packages."""
#     print_header("INSTALLING METRICS PACKAGES (OPTIONAL)")
    
#     print_info("Attempting to install PESQ and STOI packages...")
#     print_info("These may fail due to C compiler requirements - fallbacks are available")
    
#     # These are optional - failures are OK
#     packages = [
#         ("pesq", "PESQ"),
#         ("pystoi", "pySTOI"),
#     ]
    
#     for spec, name in packages:
#         success = install_package(spec, name, timeout=180)
#         if not success:
#             print_warning(f"{name} installation failed - will use Python approximation")
    
#     return True  # Always return True since these are optional

# def install_gui_packages():
#     """Install GUI packages."""
#     print_header("INSTALLING GUI PACKAGES")
    
#     # Try PyQt5 first
#     print_info("Attempting to install PyQt5 (preferred GUI backend)...")
#     success = install_package("PyQt5>=5.15.0,<5.16.0", "PyQt5", timeout=300)
    
#     if success:
#         print_success("PyQt5 installed - GUI will use PyQt5")
#         return True
#     else:
#         print_warning("PyQt5 installation failed")
#         print_info("Tkinter will be used as fallback (usually pre-installed)")
        
#         # Check if tkinter is available
#         verify_cmd = [sys.executable, "-c", "import tkinter; print('OK')"]
#         success, stdout, stderr = run_command(verify_cmd, timeout=10)
        
#         if success:
#             print_success("Tkinter is available - GUI will work")
#             return True
#         else:
#             print_warning("Tkinter not available either")
#             print_info("GUI may not work, but CLI modes will function")
#             return False

# def verify_installation():
#     """Verify all packages are importable."""
#     print_header("VERIFYING INSTALLATION")
    
#     packages_to_check = [
#         ("numpy", "NumPy"),
#         ("scipy", "SciPy"),
#         ("matplotlib", "Matplotlib"),
#         ("sounddevice", "sounddevice"),
#         ("soundfile", "soundfile"),
#         ("librosa", "librosa"),
#         ("tqdm", "tqdm"),
#         ("aiohttp", "aiohttp"),
#     ]
    
#     pytorch_packages = [
#         ("torch", "PyTorch"),
#         ("torchaudio", "TorchAudio"),
#     ]
    
#     optional_packages = [
#         ("PyQt5", "PyQt5"),
#         ("tkinter", "Tkinter"),
#         ("pesq", "PESQ"),
#         ("pystoi", "pySTOI"),
#         ("aiortc", "aiortc"),
#     ]
    
#     print_info("Checking required packages:")
#     all_good = True
    
#     for module, name in packages_to_check:
#         cmd = [sys.executable, "-c", f"import {module}"]
#         success, _, _ = run_command(cmd, timeout=10)
        
#         if success:
#             print_success(f"  {name}")
#         else:
#             print_error(f"  {name}")
#             all_good = False
    
#     # Check PyTorch separately (might need import workaround)
#     print_info("\nChecking PyTorch packages:")
#     for module, name in pytorch_packages:
#         # Try direct import first
#         try:
#             __import__(module)
#             print_success(f"  {name} (direct import)")
#         except ImportError:
#             # Try subprocess
#             cmd = [sys.executable, "-c", f"import {module}"]
#             success, _, _ = run_command(cmd, timeout=10)
            
#             if success:
#                 print_warning(f"  {name} (works in subprocess, may need terminal restart)")
#             else:
#                 print_error(f"  {name}")
#                 all_good = False
    
#     print_info("\nChecking optional packages:")
#     for module, name in optional_packages:
#         cmd = [sys.executable, "-c", f"import {module}"]
#         success, _, _ = run_command(cmd, timeout=10)
        
#         if success:
#             print_success(f"  {name}")
#         else:
#             print_warning(f"  {name} (optional - fallback available)")
    
#     return all_good

# def create_test_script():
#     """Create a test script to verify GPU functionality."""
#     print_header("CREATING TEST SCRIPT")
    
#     # Use only ASCII characters to avoid encoding issues
#     test_script = """#!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# import sys
# import torch
# import numpy as np

# print("=" * 80)
# print("PYTORCH GPU TEST")
# print("=" * 80)

# print(f"\\nPython version: {sys.version}")
# print(f"Python executable: {sys.executable}")
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")

# if torch.cuda.is_available():
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"cuDNN version: {torch.backends.cudnn.version()}")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
    
#     for i in range(torch.cuda.device_count()):
#         print(f"\\nGPU {i}: {torch.cuda.get_device_name(i)}")
        
#         # Get memory info
#         total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
#         print(f"  Total memory: {total_mem:.2f} GB")
        
#         # Test tensor creation
#         try:
#             device = torch.device(f'cuda:{i}')
#             x = torch.randn(1000, 1000, device=device)
#             y = torch.randn(1000, 1000, device=device)
#             z = torch.matmul(x, y)
#             print(f"  [OK] Tensor operations working")
            
#             # Check memory usage
#             allocated = torch.cuda.memory_allocated(i) / 1024**3
#             reserved = torch.cuda.memory_reserved(i) / 1024**3
#             print(f"  Memory allocated: {allocated:.3f} GB")
#             print(f"  Memory reserved: {reserved:.3f} GB")
            
#             del x, y, z
#             torch.cuda.empty_cache()
            
#         except Exception as e:
#             print(f"  [ERROR] Error testing GPU: {e}")
# else:
#     print("\\nNo CUDA GPU available - will run on CPU")
#     print("Testing CPU tensor operations...")
#     try:
#         x = torch.randn(100, 100)
#         y = torch.randn(100, 100)
#         z = torch.matmul(x, y)
#         print("[OK] CPU tensor operations working")
#     except Exception as e:
#         print(f"[ERROR] CPU test failed: {e}")

# print("\\n" + "=" * 80)
# print("Testing complete!")
# print("=" * 80)
# """
    
#     test_file = Path("test_gpu.py")
#     try:
#         # Use UTF-8 encoding explicitly
#         with open(test_file, 'w', encoding='utf-8') as f:
#             f.write(test_script)
        
#         print_success(f"Created test script: {test_file}")
#         print_info("Run with: python test_gpu.py")
#         return True
#     except Exception as e:
#         print_error(f"Failed to create test script: {e}")
#         return False

# def create_requirements_file():
#     """Create requirements.txt file for future reference."""
#     print_info("Creating requirements.txt...")
    
#     requirements = """# Neural Audio Codec Dependencies
# # Python 3.10.11

# # PyTorch (install separately with CUDA support)
# # torch==2.0.1
# # torchaudio==2.0.2

# # Core packages
# numpy>=1.21.0,<2.0.0
# scipy>=1.7.0,<1.12.0
# matplotlib>=3.3.0,<3.8.0

# # Audio processing
# sounddevice>=0.4.0
# soundfile>=0.10.0
# librosa>=0.9.0,<0.11.0

# # Utilities
# tqdm>=4.60.0
# Pillow>=8.0.0

# # Networking
# aiohttp>=3.8.0,<4.0.0

# # Optional: Metrics (may require C compiler)
# pesq
# pystoi

# # Optional: GUI
# PyQt5>=5.15.0,<5.16.0

# # Optional: WebRTC
# aiortc>=1.3.0
# """
    
#     try:
#         with open("requirements.txt", 'w', encoding='utf-8') as f:
#             f.write(requirements)
#         print_success("Created requirements.txt")
#         return True
#     except Exception as e:
#         print_warning(f"Failed to create requirements.txt: {e}")
#         return False

# def print_summary(results):
#     """Print installation summary."""
#     print_header("INSTALLATION SUMMARY")
    
#     print("Results:")
#     for category, success in results.items():
#         if success:
#             print_success(f"  {category}")
#         else:
#             print_error(f"  {category}")
    
#     all_critical_ok = all([
#         results.get('Python Version', False),
#         results.get('pip', False),
#         results.get('PyTorch', False),
#         results.get('Core Packages', False),
#         results.get('Audio Packages', False),
#     ])
    
#     print("\n" + "=" * 80)
#     if all_critical_ok:
#         print_success("INSTALLATION SUCCESSFUL!")
#         print("\nNext steps:")
#         print("  1. Close and reopen your terminal (to refresh PATH)")
#         print("  2. Test GPU: python test_gpu.py")
#         print("  3. Run the audio codec:")
#         print("     python app.py                    # Launch GUI")
#         print("     python app.py --mode selftest    # Run self-test")
#         print("\nIf PyTorch import fails, restart your terminal first!")
#     else:
#         print_warning("INSTALLATION COMPLETED WITH WARNINGS")
#         print("\nSome packages failed to install.")
#         print("The application may still work with reduced functionality.")
#         print("Check the errors above for details.")
        
#         if not results.get('PyTorch', False):
#             print("\n" + "!" * 80)
#             print("PyTorch installation failed!")
#             print("Try manual installation:")
#             print("  pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118")
#             print("!" * 80)
    
#     print("=" * 80 + "\n")

# def main():
#     """Main installation routine."""
#     parser = argparse.ArgumentParser(
#         description="Install all dependencies for Neural Audio Codec (Python 3.10.11 + GTX 1060 6GB)"
#     )
#     parser.add_argument('--cpu-only', action='store_true',
#                        help='Install CPU-only PyTorch (skip GPU support)')
#     parser.add_argument('--cuda-11', action='store_true',
#                        help='Force CUDA 11.8 version')
#     parser.add_argument('--cuda-12', action='store_true',
#                        help='Force CUDA 12.1 version')
#     parser.add_argument('--skip-pytorch', action='store_true',
#                        help='Skip PyTorch installation (if already installed)')
#     parser.add_argument('--skip-optional', action='store_true',
#                        help='Skip optional packages (PESQ, STOI, aiortc)')
    
#     args = parser.parse_args()
    
#     print_header("NEURAL AUDIO CODEC - DEPENDENCY INSTALLER")
#     print("Optimized for Python 3.10.11 and NVIDIA GTX 1060 6GB")
#     print(f"System: {platform.system()} {platform.machine()}")
    
#     results = {}
    
#     # Step 1: Check Python version
#     results['Python Version'] = check_python_version()
#     if not results['Python Version']:
#         print_error("Python version check failed - stopping installation")
#         return 1
    
#     # Step 2: Check pip
#     results['pip'] = check_pip()
#     if not results['pip']:
#         print_error("pip check failed - stopping installation")
#         return 1
    
#     # Step 3: Detect GPU
#     gpu_info = detect_nvidia_gpu()
    
#     # Step 4: Install PyTorch
#     if not args.skip_pytorch:
#         force_cuda = None
#         if args.cuda_11:
#             force_cuda = '11'
#         elif args.cuda_12:
#             force_cuda = '12'
        
#         cuda_version = determine_cuda_version(gpu_info, args.cpu_only, force_cuda)
#         results['PyTorch'] = install_pytorch(cuda_version)
        
#         if not results['PyTorch']:
#             print_error("PyTorch installation failed")
#             print_info("You can try to install it manually and re-run with --skip-pytorch")
#             # Don't exit - continue with other packages
#     else:
#         print_info("Skipping PyTorch installation (--skip-pytorch)")
#         results['PyTorch'] = True
    
#     # Step 5: Install core packages
#     results['Core Packages'] = install_core_packages()
    
#     # Step 6: Install audio packages
#     results['Audio Packages'] = install_audio_packages()
    
#     # Step 7: Install utility packages
#     results['Utility Packages'] = install_utility_packages()
    
#     # Step 8: Install network packages
#     results['Network Packages'] = install_network_packages()
    
#     # Step 9: Install optional packages
#     if not args.skip_optional:
#         results['Metrics Packages'] = install_metrics_packages()
#         results['GUI Packages'] = install_gui_packages()
#     else:
#         print_info("Skipping optional packages (--skip-optional)")
    
#     # Step 10: Verify installation
#     results['Verification'] = verify_installation()
    
#     # Step 11: Create test script
#     create_test_script()
    
#     # Step 12: Create requirements.txt
#     create_requirements_file()
    
#     # Step 13: Print summary
#     print_summary(results)
    
#     return 0

# if __name__ == '__main__':
#     try:
#         sys.exit(main())
#     except KeyboardInterrupt:
#         print_error("\n\nInstallation interrupted by user")
#         sys.exit(1)
#     except Exception as e:
#         print_error(f"\n\nUnexpected error: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

#!/usr/bin/env python3
"""
install_dependencies.py - Comprehensive Dependency Installer for Neural Audio Codec
Optimized for Python 3.10.11 and NVIDIA GTX 1060 6GB

FIXES Windows Store Python import issues!

Usage:
    python install_dependencies.py              # Auto-detect and install
    python install_dependencies.py --fix-paths  # Fix Windows Store Python paths
    python install_dependencies.py --use-venv   # Create and use virtual environment (RECOMMENDED)
"""

import sys
import subprocess
import platform
import os
import argparse
import re
import site
from pathlib import Path

# ANSI color codes for pretty output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
    print(f"{text}")
    print(f"{'=' * 80}{Colors.ENDC}\n")

def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}[OK] {text}{Colors.ENDC}")

def print_warning(text):
    """Print warning message."""
    print(f"{Colors.WARNING}[WARNING] {text}{Colors.ENDC}")

def print_error(text):
    """Print error message."""
    print(f"{Colors.FAIL}[ERROR] {text}{Colors.ENDC}")

def print_info(text):
    """Print info message."""
    print(f"{Colors.OKCYAN}[INFO] {text}{Colors.ENDC}")

def is_windows_store_python():
    """Detect if running Windows Store Python."""
    return (
        platform.system() == 'Windows' and
        'WindowsApps' in sys.executable
    )

def diagnose_python_environment():
    """Diagnose Python environment and potential issues."""
    print_header("PYTHON ENVIRONMENT DIAGNOSIS")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    
    # Check if Windows Store Python
    if is_windows_store_python():
        print_warning("Windows Store Python detected!")
        print_warning("This version has known import issues")
        print_info("RECOMMENDED: Use virtual environment (--use-venv) or regular Python")
    
    # Print sys.path
    print("\nPython module search paths (sys.path):")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    
    # Print site packages
    print("\nSite packages directories:")
    for path in site.getsitepackages():
        print(f"  {path}")
        if os.path.exists(path):
            print(f"    (exists)")
        else:
            print(f"    (does not exist)")
    
    # User site packages
    user_site = site.getusersitepackages()
    print(f"\nUser site packages: {user_site}")
    if os.path.exists(user_site):
        print(f"  (exists)")
    else:
        print(f"  (does not exist)")
    
    return is_windows_store_python()

def fix_windows_store_paths():
    """Add Windows Store Python paths to sys.path."""
    print_header("FIXING WINDOWS STORE PYTHON PATHS")
    
    if not is_windows_store_python():
        print_info("Not Windows Store Python - no fix needed")
        return False
    
    # Add user site packages to sys.path
    user_site = site.getusersitepackages()
    if user_site not in sys.path and os.path.exists(user_site):
        sys.path.insert(0, user_site)
        print_success(f"Added to sys.path: {user_site}")
    
    # Add site packages
    for site_pkg in site.getsitepackages():
        if site_pkg not in sys.path and os.path.exists(site_pkg):
            sys.path.insert(0, site_pkg)
            print_success(f"Added to sys.path: {site_pkg}")
    
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    print_header("CREATING VIRTUAL ENVIRONMENT")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print_warning(f"Virtual environment already exists at {venv_path}")
        response = input("Delete and recreate? (y/n): ").strip().lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
        else:
            print_info("Using existing virtual environment")
            return venv_path
    
    print_info("Creating virtual environment...")
    cmd = [sys.executable, "-m", "venv", str(venv_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print_success(f"Virtual environment created at {venv_path}")
        
        # Get path to Python in venv
        if platform.system() == 'Windows':
            venv_python = venv_path / "Scripts" / "python.exe"
            activate_script = venv_path / "Scripts" / "activate.bat"
        else:
            venv_python = venv_path / "bin" / "python"
            activate_script = venv_path / "bin" / "activate"
        
        print_info("\nTo activate this virtual environment:")
        if platform.system() == 'Windows':
            print(f"  venv\\Scripts\\activate")
        else:
            print(f"  source venv/bin/activate")
        
        print_info("\nRestart installer with virtual environment:")
        print(f"  {venv_python} install_dependencies.py")
        
        return venv_path
    else:
        print_error("Failed to create virtual environment")
        print_error(result.stderr)
        return None

def run_command(cmd, timeout=300, capture_output=True):
    """Run a shell command and return result."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=False,
            encoding='utf-8',
            errors='replace'
        )
        return (result.returncode == 0, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "", "Command timed out")
    except Exception as e:
        return (False, "", str(e))

def check_python_version():
    """Check if Python version is compatible."""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Detected Python version: {version_str}")
    print(f"Running on: {platform.system()} {platform.machine()}")
    print(f"Python executable: {sys.executable}")
    
    if version.major != 3:
        print_error(f"Python 3.x required, but found Python {version.major}")
        return False
    
    if version.minor < 8:
        print_error(f"Python 3.8+ required, but found Python {version_str}")
        return False
    
    if version.minor > 11:
        print_warning(f"Python {version_str} detected (script optimized for 3.10.11)")
        print_warning("Some packages may have compatibility issues")
    
    if version.minor == 10 and version.micro == 11:
        print_success(f"Perfect! Python {version_str} is the target version")
    else:
        print_warning(f"Python {version_str} detected (recommended: 3.10.11)")
        print_info("Should work, but some differences may occur")
    
    return True

def check_pip():
    """Ensure pip is installed and up to date."""
    print_header("CHECKING PIP")
    
    # Check if pip is available
    success, stdout, stderr = run_command([sys.executable, "-m", "pip", "--version"])
    
    if not success:
        print_error("pip is not available")
        print_info("Installing pip...")
        
        # Try to install pip
        success, _, _ = run_command([sys.executable, "-m", "ensurepip", "--default-pip"])
        if not success:
            print_error("Failed to install pip")
            return False
    
    print_success("pip is available")
    print(f"  {stdout.strip()}")
    
    # Upgrade pip
    print_info("Upgrading pip to latest version...")
    success, _, _ = run_command([
        sys.executable, "-m", "pip", "install", "--upgrade", "pip"
    ])
    
    if success:
        print_success("pip upgraded successfully")
    else:
        print_warning("Failed to upgrade pip, but continuing...")
    
    return True

def detect_nvidia_gpu():
    """Detect NVIDIA GPU and CUDA version."""
    print_header("DETECTING NVIDIA GPU")
    
    result = {
        'has_nvidia': False,
        'gpu_name': None,
        'cuda_version': None,
        'compute_capability': None,
        'driver_version': None
    }
    
    # Try nvidia-smi
    success, stdout, stderr = run_command(["nvidia-smi"], timeout=10)
    
    if not success:
        print_warning("nvidia-smi not found or failed")
        print_info("No NVIDIA GPU detected or drivers not installed")
        return result
    
    print_success("nvidia-smi found")
    
    # Parse nvidia-smi output
    lines = stdout.split('\n')
    
    # Extract driver version
    for line in lines:
        if 'Driver Version:' in line:
            match = re.search(r'Driver Version:\s+(\d+\.\d+)', line)
            if match:
                result['driver_version'] = match.group(1)
                print(f"  Driver Version: {result['driver_version']}")
        
        if 'CUDA Version:' in line:
            match = re.search(r'CUDA Version:\s+(\d+\.\d+)', line)
            if match:
                result['cuda_version'] = match.group(1)
                print(f"  CUDA Version: {result['cuda_version']}")
    
    # Get GPU name
    success, stdout, stderr = run_command([
        "nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"
    ], timeout=10)
    
    if success:
        lines = stdout.strip().split('\n')
        if lines:
            parts = lines[0].split(',')
            if len(parts) >= 1:
                result['gpu_name'] = parts[0].strip()
                print(f"  GPU Name: {result['gpu_name']}")
            if len(parts) >= 2:
                result['compute_capability'] = parts[1].strip()
                print(f"  Compute Capability: {result['compute_capability']}")
        
        result['has_nvidia'] = True
        
        # Check if it's GTX 1060
        if 'GTX 1060' in result['gpu_name'] or '1060' in result['gpu_name']:
            print_success("GTX 1060 detected - perfect for this project!")
            print_info("VRAM: 6GB (make sure you have the 6GB version, not 3GB)")
        elif result['compute_capability']:
            cc_major = float(result['compute_capability'].split('.')[0]) if '.' in result['compute_capability'] else 0
            if cc_major >= 6.0:
                print_success(f"GPU detected with compute capability {result['compute_capability']}")
                print_info("This GPU should work well with the neural codec")
    
    return result

def determine_cuda_version(gpu_info, force_cpu=False, force_cuda=None):
    """Determine which CUDA version to use for PyTorch."""
    if force_cpu:
        print_info("CPU-only mode forced by user")
        return 'cpu'
    
    if force_cuda:
        cuda_map = {
            '11': 'cu118',
            '11.8': 'cu118',
            '118': 'cu118',
            '12': 'cu121',
            '12.1': 'cu121',
            '121': 'cu121',
        }
        cuda_version = cuda_map.get(force_cuda, force_cuda)
        print_info(f"CUDA version forced by user: {cuda_version}")
        return cuda_version
    
    if not gpu_info['has_nvidia']:
        print_info("No NVIDIA GPU detected, using CPU version")
        return 'cpu'
    
    # GTX 1060 works best with CUDA 11.8
    if gpu_info['compute_capability']:
        cc = float(gpu_info['compute_capability'])
        if cc >= 6.0:
            print_info("Using CUDA 11.8 (best compatibility for GTX 1060)")
            return 'cu118'
    
    print_info("Using CUDA 11.8 (default for detected NVIDIA GPU)")
    return 'cu118'

def install_pytorch(cuda_version='cu118'):
    """Install PyTorch with appropriate CUDA support."""
    print_header(f"INSTALLING PYTORCH (CUDA version: {cuda_version})")
    
    pytorch_version = "2.0.1"
    torchaudio_version = "2.0.2"
    
    if cuda_version == 'cpu':
        print_info(f"Installing PyTorch {pytorch_version} (CPU-only)...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            f"torch=={pytorch_version}",
            f"torchaudio=={torchaudio_version}",
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ]
    elif cuda_version == 'cu118':
        print_info(f"Installing PyTorch {pytorch_version} with CUDA 11.8...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            f"torch=={pytorch_version}",
            f"torchaudio=={torchaudio_version}",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
    elif cuda_version == 'cu121':
        print_info(f"Installing PyTorch {pytorch_version} with CUDA 12.1...")
        cmd = [
            sys.executable, "-m", "pip", "install",
            f"torch=={pytorch_version}",
            f"torchaudio=={torchaudio_version}",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ]
    else:
        print_error(f"Unknown CUDA version: {cuda_version}")
        return False
    
    print_info("This may take several minutes...")
    print_info("Command: " + " ".join(cmd))
    
    success, stdout, stderr = run_command(cmd, timeout=600)
    
    if success:
        print_success("PyTorch installed successfully")
        
        # Fix paths before verification
        if is_windows_store_python():
            fix_windows_store_paths()
        
        # Verify installation
        print_info("Verifying PyTorch installation...")
        
        try:
            import torch
            print_success("PyTorch verification:")
            print(f"    PyTorch {torch.__version__}")
            print(f"    CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"    CUDA version: {torch.version.cuda}")
                print(f"    Device: {torch.cuda.get_device_name(0)}")
            else:
                print(f"    Device: CPU")
            return True
        except ImportError as e:
            print_warning(f"Direct import failed: {e}")
            print_warning("PyTorch is installed but may need terminal restart")
            return True
    else:
        print_error("PyTorch installation failed")
        print_error(f"Error: {stderr[:500]}")
        return False

def install_package(package_spec, package_name=None, timeout=120):
    """Install a Python package."""
    if package_name is None:
        package_name = package_spec.split('[')[0].split('=')[0].split('>')[0].split('<')[0]
    
    print(f"\n  Installing {package_name}...", end=" ", flush=True)
    
    cmd = [sys.executable, "-m", "pip", "install", package_spec, "--user"]
    success, stdout, stderr = run_command(cmd, timeout=timeout)
    
    if success:
        print_success("OK")
        return True
    else:
        print_error("FAILED")
        if stderr:
            print_error(f"    Error: {stderr[:200]}")
        return False

def install_core_packages():
    """Install core required packages."""
    print_header("INSTALLING CORE PACKAGES")
    
    packages = [
        ("numpy>=1.21.0,<2.0.0", "NumPy"),
        ("scipy>=1.7.0,<1.12.0", "SciPy"),
        ("matplotlib>=3.3.0,<3.8.0", "Matplotlib"),
    ]
    
    results = {}
    for spec, name in packages:
        results[name] = install_package(spec, name)
    
    return all(results.values())

def install_audio_packages():
    """Install audio processing packages."""
    print_header("INSTALLING AUDIO PACKAGES")
    
    packages = [
        ("sounddevice>=0.4.0", "sounddevice"),
        ("soundfile>=0.10.0", "soundfile"),
        ("librosa>=0.9.0,<0.11.0", "librosa"),
    ]
    
    results = {}
    for spec, name in packages:
        results[name] = install_package(spec, name, timeout=180)
    
    return all(results.values())

def install_utility_packages():
    """Install utility packages."""
    print_header("INSTALLING UTILITY PACKAGES")
    
    packages = [
        ("tqdm>=4.60.0", "tqdm"),
        ("Pillow>=8.0.0", "Pillow"),
    ]
    
    results = {}
    for spec, name in packages:
        results[name] = install_package(spec, name)
    
    return all(results.values())

def install_network_packages():
    """Install networking packages."""
    print_header("INSTALLING NETWORK PACKAGES")
    
    packages = [
        ("aiohttp>=3.8.0,<4.0.0", "aiohttp"),
    ]
    
    results = {}
    for spec, name in packages:
        results[name] = install_package(spec, name)
    
    # Optional
    print_info("Attempting to install optional package: aiortc")
    install_package("aiortc>=1.3.0", "aiortc")
    
    return all(results.values())

def install_metrics_packages():
    """Install audio metrics packages."""
    print_header("INSTALLING METRICS PACKAGES (OPTIONAL)")
    
    print_info("Attempting to install PESQ and STOI packages...")
    print_info("These may fail due to C compiler requirements - fallbacks are available")
    
    packages = [
        ("pesq", "PESQ"),
        ("pystoi", "pySTOI"),
    ]
    
    for spec, name in packages:
        success = install_package(spec, name, timeout=180)
        if not success:
            print_warning(f"{name} installation failed - will use Python approximation")
    
    return True

def install_gui_packages():
    """Install GUI packages."""
    print_header("INSTALLING GUI PACKAGES")
    
    # Try PyQt5 first (preferred for the main app.py)
    print_info("Attempting to install PyQt5 (preferred GUI backend)...")
    success = install_package("PyQt5>=5.15.0,<5.16.0", "PyQt5", timeout=300)
    
    if success:
        print_success("PyQt5 installed - GUI will use PyQt5")
    else:
        print_warning("PyQt5 installation failed - will try alternatives")
    
    # Note about PySimpleGUI
    print_info("\nNote: PySimpleGUI is deprecated. If you need it, install FreeSimpleGUI instead:")
    print_info("  pip install FreeSimpleGUI")
    
    # Check tkinter
    verify_cmd = [sys.executable, "-c", "import tkinter; print('OK')"]
    tk_success, stdout, stderr = run_command(verify_cmd, timeout=10)
    
    if tk_success:
        print_success("Tkinter is available as fallback")
    else:
        print_warning("Tkinter not available")
    
    return success or tk_success

def verify_installation():
    """Verify all packages are importable."""
    print_header("VERIFYING INSTALLATION")
    
    # Fix paths first
    if is_windows_store_python():
        fix_windows_store_paths()
    
    packages_to_check = [
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("matplotlib", "Matplotlib"),
        ("sounddevice", "sounddevice"),
        ("soundfile", "soundfile"),
        ("librosa", "librosa"),
        ("tqdm", "tqdm"),
        ("aiohttp", "aiohttp"),
    ]
    
    pytorch_packages = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
    ]
    
    optional_packages = [
        ("PyQt5", "PyQt5"),
        ("tkinter", "Tkinter"),
        ("pesq", "PESQ"),
        ("pystoi", "pySTOI"),
        ("aiortc", "aiortc"),
    ]
    
    print_info("Checking required packages:")
    all_good = True
    
    for module, name in packages_to_check:
        try:
            __import__(module)
            print_success(f"  {name}")
        except ImportError:
            print_error(f"  {name}")
            all_good = False
    
    print_info("\nChecking PyTorch packages:")
    for module, name in pytorch_packages:
        try:
            __import__(module)
            print_success(f"  {name}")
        except ImportError:
            print_error(f"  {name}")
            all_good = False
    
    print_info("\nChecking optional packages:")
    for module, name in optional_packages:
        try:
            __import__(module)
            print_success(f"  {name}")
        except ImportError:
            print_warning(f"  {name} (optional)")
    
    return all_good

def create_test_script():
    """Create a test script to verify GPU functionality."""
    print_header("CREATING TEST SCRIPT")
    
    test_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import torch
import numpy as np

print("=" * 80)
print("PYTORCH GPU TEST")
print("=" * 80)

print(f"\\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\\nGPU {i}: {torch.cuda.get_device_name(i)}")
        
        total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  Total memory: {total_mem:.2f} GB")
        
        try:
            device = torch.device(f'cuda:{i}')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            print(f"  [OK] Tensor operations working")
            
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Memory allocated: {allocated:.3f} GB")
            print(f"  Memory reserved: {reserved:.3f} GB")
            
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  [ERROR] {e}")
else:
    print("\\nNo CUDA GPU available - will run on CPU")

print("\\n" + "=" * 80)
print("Testing complete!")
print("=" * 80)
"""
    
    try:
        with open("test_gpu.py", 'w', encoding='utf-8') as f:
            f.write(test_script)
        print_success("Created test_gpu.py")
        return True
    except Exception as e:
        print_error(f"Failed to create test script: {e}")
        return False

def create_activation_helper():
    """Create helper scripts for Windows Store Python."""
    print_header("CREATING HELPER SCRIPTS")
    
    if not is_windows_store_python():
        print_info("Not Windows Store Python - skipping helper scripts")
        return True
    
    # Create a batch file to set PYTHONPATH
    batch_content = f"""@echo off
REM Helper script for Windows Store Python
echo Setting up Python environment...

set PYTHONPATH=%LOCALAPPDATA%\\Packages\\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python310\\site-packages;%PYTHONPATH%

echo PYTHONPATH set to:
echo %PYTHONPATH%

echo.
echo You can now run:
echo   python app.py
echo   python test_gpu.py
echo.

cmd /k
"""
    
    try:
        with open("run_with_fixed_paths.bat", 'w') as f:
            f.write(batch_content)
        print_success("Created run_with_fixed_paths.bat")
        print_info("Use this if you have import errors:")
        print_info("  run_with_fixed_paths.bat")
        return True
    except Exception as e:
        print_warning(f"Failed to create helper script: {e}")
        return False

def print_summary(results, use_venv_recommended=False):
    """Print installation summary."""
    print_header("INSTALLATION SUMMARY")
    
    print("Results:")
    for category, success in results.items():
        if success:
            print_success(f"  {category}")
        else:
            print_error(f"  {category}")
    
    all_critical_ok = all([
        results.get('Python Version', False),
        results.get('pip', False),
        results.get('PyTorch', False),
        results.get('Core Packages', False),
        results.get('Audio Packages', False),
    ])
    
    print("\n" + "=" * 80)
    if all_critical_ok:
        print_success("INSTALLATION SUCCESSFUL!")
        
        if use_venv_recommended:
            print("\n" + "!" * 80)
            print_warning("WINDOWS STORE PYTHON DETECTED!")
            print_warning("For best results, use a virtual environment:")
            print("\n  1. python -m venv venv")
            print("  2. venv\\Scripts\\activate")
            print("  3. python install_dependencies.py")
            print("!" * 80)
        
        print("\nNext steps:")
        print("  1. Test GPU: python test_gpu.py")
        print("  2. Run the codec:")
        print("     python app.py                    # Launch GUI")
        print("     python app.py --mode selftest    # Run self-test")
        
        if is_windows_store_python():
            print("\n  If you get import errors, run:")
            print("     run_with_fixed_paths.bat")
        
    else:
        print_warning("INSTALLATION COMPLETED WITH WARNINGS")
        print("\nSome packages failed to install.")
        
        if use_venv_recommended:
            print("\n" + "!" * 80)
            print_error("WINDOWS STORE PYTHON HAS KNOWN ISSUES")
            print_error("STRONGLY RECOMMENDED: Use virtual environment")
            print("\nRun: python install_dependencies.py --use-venv")
            print("!" * 80)
    
    print("=" * 80 + "\n")

def main():
    """Main installation routine."""
    parser = argparse.ArgumentParser(
        description="Install all dependencies for Neural Audio Codec"
    )
    parser.add_argument('--cpu-only', action='store_true',
                       help='Install CPU-only PyTorch')
    parser.add_argument('--cuda-11', action='store_true',
                       help='Force CUDA 11.8')
    parser.add_argument('--cuda-12', action='store_true',
                       help='Force CUDA 12.1')
    parser.add_argument('--skip-pytorch', action='store_true',
                       help='Skip PyTorch installation')
    parser.add_argument('--skip-optional', action='store_true',
                       help='Skip optional packages')
    parser.add_argument('--fix-paths', action='store_true',
                       help='Fix Windows Store Python paths')
    parser.add_argument('--use-venv', action='store_true',
                       help='Create and recommend virtual environment')
    parser.add_argument('--diagnose', action='store_true',
                       help='Diagnose Python environment and exit')
    
    args = parser.parse_args()
    
    # Diagnose mode
    if args.diagnose:
        diagnose_python_environment()
        return 0
    
    # Virtual environment mode
    if args.use_venv:
        venv_path = create_virtual_environment()
        if venv_path:
            print_info("\nVirtual environment created!")
            print_info("Activate it and re-run this script:")
            if platform.system() == 'Windows':
                print(f"\n  venv\\Scripts\\activate")
            else:
                print(f"\n  source venv/bin/activate")
            print(f"  python install_dependencies.py\n")
        return 0
    
    # Path fix mode
    if args.fix_paths:
        fix_windows_store_paths()
        return 0
    
    print_header("NEURAL AUDIO CODEC - DEPENDENCY INSTALLER")
    print("Optimized for Python 3.10.11 and NVIDIA GTX 1060 6GB")
    
    # Diagnose environment
    is_winstore = diagnose_python_environment()
    
    results = {}
    
    # Check Python version
    results['Python Version'] = check_python_version()
    if not results['Python Version']:
        return 1
    
    # Check pip
    results['pip'] = check_pip()
    if not results['pip']:
        return 1
    
    # Detect GPU
    gpu_info = detect_nvidia_gpu()
    
    # Install PyTorch
    if not args.skip_pytorch:
        force_cuda = None
        if args.cuda_11:
            force_cuda = '11'
        elif args.cuda_12:
            force_cuda = '12'
        
        cuda_version = determine_cuda_version(gpu_info, args.cpu_only, force_cuda)
        results['PyTorch'] = install_pytorch(cuda_version)
    else:
        results['PyTorch'] = True
    
    # Install other packages
    results['Core Packages'] = install_core_packages()
    results['Audio Packages'] = install_audio_packages()
    results['Utility Packages'] = install_utility_packages()
    results['Network Packages'] = install_network_packages()
    
    if not args.skip_optional:
        results['Metrics Packages'] = install_metrics_packages()
        results['GUI Packages'] = install_gui_packages()
    
    # Verify
    results['Verification'] = verify_installation()
    
    # Create helper files
    create_test_script()
    if is_winstore:
        create_activation_helper()
    
    # Summary
    print_summary(results, is_winstore)
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_error("\n\nInstallation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)