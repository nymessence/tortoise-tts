import sys
import platform
import subprocess
from packaging.version import Version

def get_torch_version():
    """
    Dynamically gets the installed PyTorch version.
    """
    try:
        import torch
        return Version(torch.__version__)
    except ImportError:
        print("Error: PyTorch not found. Please install a compatible version first.", file=sys.stderr)
        return None

def get_python_version():
    """
    Gets the current Python version in a simplified format (e.g., 'cp37').
    """
    return f"cp{sys.version_info.major}{sys.version_info.minor}"

def get_machine_arch():
    """
    Gets the machine architecture (e.g., 'x86_64').
    """
    return platform.machine()

def get_xla_wheel_url(torch_version, python_version, machine_arch):
    """
    Constructs the correct torch_xla wheel URL based on detected versions and architecture.
    """
    # Dynamically determine the XLA version based on the PyTorch version.
    xla_version = f"{torch_version.major}.{torch_version.minor}"
    
    base_url = "https://storage.googleapis.com/tpu-pytorch/wheels"
    url = (
        f"{base_url}/torch_xla-{xla_version}-{python_version}-"
        f"{python_version}m-linux_{machine_arch}.whl"
    )
    return url
