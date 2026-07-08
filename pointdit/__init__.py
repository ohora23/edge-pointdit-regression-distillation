"""PointDiT — point-map diffusion transformer for (transparent-object) monocular geometry.

RK3588-portable reproduction scaffold. See .omc/research/pointdit-rk3588/REPORT.md for the
methodology this implements.
"""
from .config import ModelConfig, RESOLUTIONS

__all__ = ["ModelConfig", "RESOLUTIONS"]
__version__ = "0.1.0"
