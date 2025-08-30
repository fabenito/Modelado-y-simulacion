"""
Métodos de integración numérica.
Implementaciones completas de Newton-Cotes y técnicas adaptativas.
"""

from .base import (
    IntegrationMethod,
    NewtonCotesMethod, 
    AdaptiveMethod,
    IntegrationPoint,
    IntegrationResult
)

from .rectangle import RectangleMethod, regla_rectangulo
from .trapezoidal import TrapezoidalMethod, regla_trapezoidal
from .simpson import Simpson13Method, Simpson38Method, regla_simpson_13, regla_simpson_38
from .boole import BooleMethod, regla_boole
from .adaptive import AdaptiveSimpsonMethod, integracion_adaptativa_simpson

__all__ = [
    # Clases base
    'IntegrationMethod',
    'NewtonCotesMethod',
    'AdaptiveMethod',
    'IntegrationPoint', 
    'IntegrationResult',
    
    # Métodos específicos
    'RectangleMethod',
    'TrapezoidalMethod', 
    'Simpson13Method',
    'Simpson38Method',
    'BooleMethod',
    'AdaptiveSimpsonMethod',
    
    # Funciones de compatibilidad
    'regla_rectangulo',
    'regla_trapezoidal',
    'regla_simpson_13',
    'regla_simpson_38', 
    'regla_boole',
    'integracion_adaptativa_simpson'
]


def get_all_methods():
    """
    Retorna instancias de todos los métodos disponibles.
    
    Returns:
        Dict con nombre -> instancia del método
    """
    methods = {
        'rectangulo': RectangleMethod(),
        'trapezoidal': TrapezoidalMethod(),
        'simpson_13': Simpson13Method(),
        'simpson_38': Simpson38Method(),
        'boole': BooleMethod(),
        'adaptativo': AdaptiveSimpsonMethod()
    }
    return methods


def get_method_by_name(name: str) -> IntegrationMethod:
    """
    Obtiene un método por su nombre.
    
    Args:
        name: Nombre del método
        
    Returns:
        Instancia del método solicitado
        
    Raises:
        ValueError: Si el método no existe
    """
    methods = get_all_methods()
    
    if name.lower() not in methods:
        available = ', '.join(methods.keys())
        raise ValueError(f"Método '{name}' no disponible. Disponibles: {available}")
    
    return methods[name.lower()]


def compare_methods(func, a: float, b: float, n: int = 10, 
                   exact_value: float = None) -> dict:
    """
    Compara todos los métodos de integración disponibles.
    
    Args:
        func: Función a integrar
        a, b: Límites de integración
        n: Número de subdivisiones (ajustado automáticamente según método)
        exact_value: Valor exacto de la integral (opcional)
        
    Returns:
        Dict con resultados comparativos
    """
    results = {}
    methods = get_all_methods()
    
    for name, method in methods.items():
        try:
            # Ajustar n según requerimientos del método
            if name == 'simpson_13' and n % 2 != 0:
                n_adjusted = n + 1
            elif name == 'simpson_38' and n % 3 != 0:
                n_adjusted = ((n // 3) + 1) * 3
            elif name == 'boole' and n % 4 != 0:
                n_adjusted = ((n // 4) + 1) * 4
            elif name == 'adaptativo':
                n_adjusted = None  # No usa n
            else:
                n_adjusted = n
            
            if name == 'adaptativo':
                result = method.integrate(func, a, b)
            else:
                result = method.integrate(func, a, b, n_adjusted)
            
            comparison = {
                'value': result.value,
                'evaluations': result.function_evaluations,
                'method': method.name,
                'n_used': n_adjusted
            }
            
            if exact_value is not None:
                comparison['absolute_error'] = abs(result.value - exact_value)
                comparison['relative_error'] = abs(result.value - exact_value) / abs(exact_value)
            
            results[name] = comparison
            
        except Exception as e:
            results[name] = {
                'error': str(e),
                'method': method.name
            }
    
    return results
