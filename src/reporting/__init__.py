"""
Reporting Module

Comprehensive reporting and analysis tools for trading model training and evaluation.
Provides classes and utilities for generating detailed training reports, performance analysis,
and visualization for ML/RL trading models.
"""

from .model_training_report import ModelTrainingReport

__version__ = "1.0.0"
__author__ = "Trading Bot Project"

# Public API
__all__ = [
    "ModelTrainingReport",
]


# Module-level convenience functions
def create_training_report(model_dir: str = 'models/rl_demo') -> ModelTrainingReport:
    """
    Create a ModelTrainingReport instance for RL training analysis.

    Args:
        model_dir: Directory containing the trained model

    Returns:
        ModelTrainingReport instance ready for analysis

    Example:
        >>> from src.reporting import create_training_report
        >>> report = create_training_report('models/my_model')
        >>> report.check_model_status()
    """
    return ModelTrainingReport(model_dir)
