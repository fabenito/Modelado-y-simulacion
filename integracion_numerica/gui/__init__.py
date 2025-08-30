"""
Componentes de interfaz gráfica para el simulador de integración numérica.
"""

from .main_window import MainWindow
from .results import ResultDisplay
from .visualization import IntegrationVisualization
from .formulas import FormulaDisplay

__all__ = [
    'MainWindow',
    'ResultDisplay', 
    'IntegrationVisualization',
    'FormulaDisplay'
]
