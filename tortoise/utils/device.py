import torch
import logging

try:
    # Attempt to import torch_xla for TPU support
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None

def get_device_name():
    """
    Returns the appropriate device based on available hardware.
    Checks for TPU, then GPU, then defaults to CPU.
    """
    if xm is not None and len(xm._get_xla_devices()) > 0:
        device = xm.xla_device()
        logging.info(f"Using TPU device: {device}")
        return device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info("Using GPU device.")
        return device
    device = torch.device('cpu')
    logging.info("Using CPU device.")
    return device
