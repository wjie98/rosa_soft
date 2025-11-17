import torch

import warnings
from packaging.version import parse as parse_version

import os
import sys
import platform

try:
    import psutil
except ImportError:
    psutil = None

import logging

logger = logging.getLogger(__name__)


__all__ = [
    "setup_tf32",
    "verify_tf32_status",
    "log_environment_info",
]


def setup_tf32(enable: bool = True):
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 8:
        logger.warning("TF32 settings are only applicable to CUDA devices with Ampere architecture or newer.")
        return

    # PyTorch 2.9 introduced a new, more granular API for controlling FP32 precision.
    # See: https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    is_pytorch_2_9_or_newer = parse_version(torch.__version__) >= parse_version("2.9.0")

    if is_pytorch_2_9_or_newer:
        precision = "tf32" if enable else "ieee"
        
        torch.backends.cuda.matmul.fp32_precision = precision
        torch.backends.cudnn.fp32_precision = precision
        
        status = "enabled" if enable else "disabled (IEEE precision)"
        logger.info(f"TF32 has been {status} for CUDA matmul and cuDNN.")

    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.backends.cuda.matmul.allow_tf32 = enable
            torch.backends.cudnn.allow_tf32 = enable
            
        status = "enabled" if enable else "disabled"
        logger.info(f"TF32 has been {status} for CUDA matmul and cuDNN.")


def verify_tf32_status():
    if not torch.cuda.is_available() or not torch.cuda.get_device_capability()[0] >= 8:
        return
        
    is_pytorch_2_9_or_newer = parse_version(torch.__version__) >= parse_version("2.9.0")
    if is_pytorch_2_9_or_newer:
        matmul_precision = torch.backends.cuda.matmul.fp32_precision
        cudnn_precision = torch.backends.cudnn.fp32_precision
        logger.info(f"torch.backends.cuda.matmul.fp32_precision = '{matmul_precision}'")
        logger.info(f"torch.backends.cudnn.fp32_precision = '{cudnn_precision}'")
    else:
        matmul_allowed = torch.backends.cuda.matmul.allow_tf32
        cudnn_allowed = torch.backends.cudnn.allow_tf32
        logger.info(f"torch.backends.cuda.matmul.allow_tf32 = {matmul_allowed}")
        logger.info(f"torch.backends.cudnn.allow_tf32 = {cudnn_allowed}")


def log_environment_info():
    logger.info("--- PyTorch & CUDA Setup ---")
    logger.info(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")

    if cuda_available:
        logger.info(f"CUDA Version (linked with PyTorch): {torch.version.cuda}")
        try:
            # torch.cuda.driver_version is available in newer PyTorch versions
            logger.info(f"NVIDIA Driver Version: {torch.cuda.driver_version}")
        except AttributeError:
            logger.info("NVIDIA Driver Version: Not available via torch.cuda.driver_version (use nvidia-smi).")
        
        cudnn_version = torch.backends.cudnn.version()
        logger.info(f"cuDNN Version: {cudnn_version}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            device_properties = torch.cuda.get_device_properties(i)
            total_memory_gb = device_properties.total_memory / (1024**3)
            logger.info(
                f"  - GPU {i}: {device_properties.name} | "
                f"Compute Capability: {device_properties.major}.{device_properties.minor} | "
                f"Total Memory: {total_memory_gb:.2f} GB"
            )
        
        logger.info(f"cuDNN Benchmark enabled: {torch.backends.cudnn.benchmark}")
        logger.info(f"cuDNN Deterministic enabled: {torch.backends.cudnn.deterministic}")
    
    else:
        logger.info("CUDA is not available. Training will run on CPU.")

    logger.info("\n" + "--- System & Hardware ---")
    logger.info(f"Operating System: {platform.platform()}")
    logger.info(f"Python Version: {sys.version.split()[0]}")
    logger.info(f"Number of CPU Cores: {os.cpu_count()}")

    if psutil:
        ram = psutil.virtual_memory()
        total_ram_gb = ram.total / (1024**3)
        logger.info(f"Total System RAM: {total_ram_gb:.2f} GB")
    else:
        logger.warning("`psutil` library not found. RAM information is unavailable. "
                       "Install with `pip install psutil` for more details.")

    logger.info(f"PyTorch CPU Threads: {torch.get_num_threads()}")
