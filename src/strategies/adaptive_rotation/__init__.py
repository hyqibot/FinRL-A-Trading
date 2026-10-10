"""
Adaptive Multi-Asset Rotation Strategy
Version: v1.2.1

A research-grade, walk-forward-safe multi-asset allocation strategy.
"""

__version__ = "1.2.1"

# Main engine will be imported here after implementation
from .adaptive_rotation_engine import AdaptiveRotationEngine

__all__ = ["__version__"]
