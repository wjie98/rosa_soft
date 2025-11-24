import torch

import warnings
from packaging.version import parse as parse_version

import logging

logger = logging.getLogger(__name__)


__all__ = [
    "setup_tf32",
    "verify_tf32_status",
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
    
    verify_tf32_status()


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
