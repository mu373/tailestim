"""Estimators for tail index estimation."""

from .base import BaseTailEstimator
from .hill import HillEstimator
from .moments import MomentsEstimator
from .kernel import KernelTypeEstimator
from .pickands import PickandsEstimator
from .smooth_hill import SmoothHillEstimator
from .estimator_set import TailEstimatorSet

__all__ = [
    'BaseTailEstimator',
    'HillEstimator',
    'MomentsEstimator',
    'KernelTypeEstimator',
    'PickandsEstimator',
    'SmoothHillEstimator',
    'TailEstimatorSet',
]
