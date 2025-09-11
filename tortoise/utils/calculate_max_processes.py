import argparse
import sys
import os

def get_max_processes(hardware_type):
    """
    Calculates the maximum number of processes to launch based on hardware.

    Args:
        hardware_type (str): The type of hardware detected ('tpu', 'gpu', or 'cpu').

    Returns:
        int: The recommended number of processes to launch.
    """
    if hardware_type == 'tpu':
        # Assuming 8 cores on the TPU. We can run multiple processes per core
        # to maximize resource utilization and memory access.
        processes_per_core = 4
        return 8 * processes_per_core
    elif hardware_type == 'gpu':
        # The shell script will already have detected the number of GPUs and
        # will loop accordingly. So, this function should just return 1
        # to ensure each instance runs on a single GPU.
        return 1
    else:
        # For CPU, we always run a single process as multi-processing is
        # not typically beneficial for this task on CPU.
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate max processes for TTS generation.")
    parser.add_argument("--hardware", type=str, required=True, choices=["cpu", "gpu", "tpu"], help="Hardware type to calculate processes for.")
    
    args = parser.parse_args()
    
    # Print the calculated value to stdout for the shell script to capture.
    print(get_max_processes(args.hardware))
