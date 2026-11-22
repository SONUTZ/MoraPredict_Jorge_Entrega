"""Motor preventivo para datos sintéticos. Sin decisiones crediticias automáticas."""
from .data import build_features, load_inputs, load_labels, DataValidationError
from .engine import MotorAlertas

__all__ = ['build_features', 'load_inputs', 'load_labels', 'DataValidationError', 'MotorAlertas']
