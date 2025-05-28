"""
Iterative IPO: Modular implementation for self-improvement research
"""

from .core.config import ExperimentConfig, IterationMetrics
from .experiment import IterativeIPO

__all__ = ['ExperimentConfig', 'IterationMetrics', 'IterativeIPO']