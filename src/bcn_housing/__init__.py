"""
Barcelona Housing ML package.

Core modules:
- config: constants and feature lists
- data: data loading and basic checks
- modeling: model builders
- evaluation: metrics
- plots: figures export
"""

from . import config, data, evaluation, modeling, plots


__all__ = ["config", "data", "modeling", "evaluation", "plots"]
__version__ = "0.1.0"
