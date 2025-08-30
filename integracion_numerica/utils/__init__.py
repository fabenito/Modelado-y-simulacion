"""
Módulo de utilidades para el simulador de integración numérica.
"""

from .expressions import (
    eval_safe_expression,
    make_safe_function, 
    validate_integration_parameters,
    format_result,
    calculate_step_size,
    generate_points,
    # Aliases para compatibilidad
    _eval_safe_expression,
    _make_safe_func
)

__all__ = [
    'eval_safe_expression',
    'make_safe_function',
    'validate_integration_parameters', 
    'format_result',
    'calculate_step_size',
    'generate_points',
    # Compatibility aliases
    '_eval_safe_expression',
    '_make_safe_func'
]
