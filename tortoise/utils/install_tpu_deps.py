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
        # If PyTorch isn't found, we can't proceed.
        print("Error: PyTorch not found. Please install a compatible version first.", file=sys.stderr)
        sys.exit(1)

def get_python_version():
    """
    Gets the current Python version in a simplified format (e.g., 'cp37').
    This format is used in the TPU wheel filenames.
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
    # It maps the PyTorch major.minor version to the corresponding XLA version.
    version_map = {
        Version("1.13"): "1.13",
        Version("2.0"): "2.0",
        Version("2.1"): "2.1"
    }
    
    xla_version = version_map.get(torch_version.major_minor)
    if not xla_version:
        raise ValueError(f"No compatible torch_xla version found for PyTorch {torch_version.major_minor}. Check the lookup table.")

    base_url = "https://storage.googleapis.com/tpu-pytorch/wheels"
    url = (
        f"{base_url}/torch_xla-{xla_version}-{python_version}-"
        f"{python_version}m-linux_{machine_arch}.whl"
    )
    return url

if __name__ == "__main__":
    try:
        # Get all the necessary version information
        torch_version = get_torch_version()
        python_version = get_python_version()
        machine_arch = get_machine_arch()
        
        # Generate the correct URL and print it to stdout.
        # The shell script will capture this output.
        url = get_xla_wheel_url(torch_version, python_version, machine_arch)
        print(url)
        
    except Exception as e:
        # Print errors to stderr so they don't interfere with the stdout output
        print(f"Error generating URL: {e}", file=sys.stderr)
        sys.exit(1)
