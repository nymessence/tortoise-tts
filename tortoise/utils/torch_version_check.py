import sys
import subprocess
import logging
import re
import itertools
from packaging.version import parse as parse_version
import requests
import os
import tempfile

# Configure logging to provide clear feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_torch_versions():
    """
    Fetches the latest versions of torch from PyPI's simple index.
    
    Returns:
        list: A sorted list of unique version strings (newest first).
    """
    torch_index_url = "https://pypi.org/simple/torch/"
    logging.info(f"Fetching torch versions from {torch_index_url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(torch_index_url, headers=headers)
        response.raise_for_status()
        
        versions = set()
        # Parse the versions from the simple index page
        for line in response.text.splitlines():
            match = re.search(r'torch-(.*?)-', line)
            if match:
                versions.add(match.group(1))
                
        # Sort versions from newest to oldest
        sorted_versions = sorted(list(versions), key=parse_version, reverse=True)
        latest_versions = sorted_versions[:5]  # Get top 5 versions
        logging.info(f"✅ Found latest 5 torch versions: {latest_versions}")
        return latest_versions
    
    except requests.exceptions.RequestException as e:
        logging.error(f"❌ Failed to fetch torch versions: {e}")
        return []

def test_combination(version):
    """
    Attempts to install and test a specific version of torch and torch_xla.
    Returns True if the combination is successful, False otherwise.
    """
    logging.info(f"  🧪 Testing combination: torch=={version}, torch_xla=={version}")
    try:
        # Step 1: Clean up previous installations
        logging.info("  -> Cleaning up previous installations...")
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torch_xla", "libtpu"],
            check=False,
            capture_output=True,
            text=True
        )

        # Step 2: Attempt to install the new versions
        logging.info("  -> Attempting new installation...")
        install_command = [
            sys.executable, "-m", "pip", "install", "--no-cache-dir",
            f"torch=={version}",
            f"torch_xla=={version}",
            "libtpu",
            "--extra-index-url", "https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla"
        ]
        
        result = subprocess.run(install_command, check=True, capture_output=True, text=True)
        logging.info(f"  ✅ Installation successful:\n{result.stdout}")

        # Step 3: Verify the installation with a simple import test and a TPU test
        logging.info("  -> Verifying with a simple import and TPU test...")
        import torch
        import torch_xla.core.xla_model as xm
        
        try:
            device = xm.xla_device()
            # Simple check to confirm the device is a TPU
            assert device.type == 'xla'
            # Perform a simple operation on the TPU to ensure it's functional
            x = torch.randn(10, 10, device=device)
            y = x.matmul(x)
            xm.mark_step()
            logging.info("  ✅ Imports and TPU operation successful. Combination is valid.")
            return True
        except Exception as e:
            logging.error(f"  ❌ TPU device test failed: {e}")
            return False

    except subprocess.CalledProcessError as e:
        logging.error(f"  ❌ Installation failed: {e.stderr}")
    except ImportError as e:
        logging.error(f"  ❌ Import failed after installation: {e}")
    except Exception as e:
        logging.error(f"  ❌ An unexpected error occurred during test: {e}")
    
    return False

def create_virtual_environment():
    """
    Creates a virtual environment in a temporary directory.
    Returns the path to the virtual environment's Python executable.
    """
    # Create a temporary directory for the virtual environment
    temp_dir = tempfile.mkdtemp(prefix="torch_xla_env_")
    venv_path = os.path.join(temp_dir, "venv")
    
    # Create the virtual environment
    logging.info(f"Creating virtual environment at {venv_path}")
    subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
    
    # Return the path to the Python executable inside the virtual environment
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")

def find_working_versions():
    """
    Iterates through version combinations of the latest versions of torch
    and returns the first working set where torch and torch_xla have the same version.
    """
    # Get torch versions from PyPI
    torch_versions = get_torch_versions()
    
    # If dynamic search fails, fall back to known good versions
    if not torch_versions:
        logging.warning("Falling back to a more robust list of stable versions due to dynamic search failure.")
        torch_versions = ["2.3.0", "2.2.0", "2.1.0", "2.0.0", "1.13.0"]
    
    # Generate combinations where torch and torch_xla have the same version
    version_combinations = torch_versions
    
    logging.info(f"Total combinations to test: {len(version_combinations)}")
    
    # Create a virtual environment for testing
    venv_python = create_virtual_environment()
    
    # Save the current Python executable for later restoration
    original_python = sys.executable
    
    # Temporarily change sys.executable to the virtual environment's Python
    sys.executable = venv_python
    
    try:
        for version in version_combinations:
            if test_combination(version):
                return version, version
    finally:
        # Restore the original Python executable
        sys.executable = original_python
    
    logging.error("\n💔 No compatible version combination found.")
    return None, None


