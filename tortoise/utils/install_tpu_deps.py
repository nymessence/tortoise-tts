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
    # This is a lookup table for common PyTorch/XLA version combinations.
    version_map = {
        (1, 13): "1.13",
        (2, 0): "2.0",
        (2, 1): "2.1"
    }
    
    # Correctly access the major and minor version numbers
    xla_version = version_map.get((torch_version.major, torch_version.minor))
    if not xla_version:
        raise ValueError(f"No compatible torch_xla version found for PyTorch {torch_version}. Check the lookup table.")

    base_url = "https://storage.googleapis.com/tpu-pytorch/wheels"
    url = (
        f"{base_url}/torch_xla-{xla_version}-{python_version}-"
        f"{python_version}m-linux_{machine_arch}.whl"
    )
    return url
