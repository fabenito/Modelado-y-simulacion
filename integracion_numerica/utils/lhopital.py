"""
Utilidades para manejar singularidades removibles usando la regla de L'Hôpital.
Detecta funciones que son indefinidas en un punto pero tienen límite finito.
"""

import ast
import math
import re
from typing import Optional, Tuple, Callable
from .expressions import SAFE_FUNCTIONS, SAFE_CONSTANTS


class LHopitalAnalyzer:
    """
    Analizador para detectar funciones compatibles con la regla de L'Hôpital
    y calcular límites en puntos de singularidad removible.
    """
    
    # Patrones comunes de funciones indeterminadas 0/0
    ZERO_OVER_ZERO_PATTERNS = [
        r'sin\(([^)]*)\)/\1',        # sin(x)/x
        r'sinh\(([^)]*)\)/\1',       # sinh(x)/x  
        r'tan\(([^)]*)\)/\1',        # tan(x)/x
        r'(1-cos\(([^)]*)\))/\2\*\*2',  # (1-cos(x))/x²
        r'(exp\(([^)]*)\)-1)/\2',    # (e^x-1)/x
        r'log\(1\+([^)]*)\)/\1',     # log(1+x)/x
    ]
    
    # Límites conocidos para patrones comunes
    KNOWN_LIMITS = {
        'sin(x)/x': 1.0,
        'sinh(x)/x': 1.0,
        'tan(x)/x': 1.0,
        '(1-cos(x))/x**2': 0.5,
        '(exp(x)-1)/x': 1.0,
        'log(1+x)/x': 1.0,
    }
    
    @staticmethod
    def is_function_undefined_at_point(func: Callable[[float], float], 
                                     x: float, epsilon: float = 1e-10) -> bool:
        """
        Verifica si una función es indefinida en un punto específico.
        
        Args:
            func: Función a evaluar
            x: Punto donde verificar
            epsilon: Tolerancia para números muy pequeños
            
        Returns:
            True si la función es indefinida en x
        """
        try:
            result = func(x)
            return not math.isfinite(result)
        except (ZeroDivisionError, ValueError, OverflowError):
            return True
        except Exception:
            return True
    
    @staticmethod
    def detect_lhopital_pattern(expression: str, x_value: float = 0.0) -> Optional[str]:
        """
        Detecta si una expresión tiene un patrón compatible con L'Hôpital.
        
        Args:
            expression: Expresión matemática como string
            x_value: Valor donde verificar la indeterminación
            
        Returns:
            Tipo de patrón detectado o None si no es compatible
        """
        # Normalizar la expresión (quitar espacios, usar ** para potencias)
        normalized = expression.replace(' ', '').replace('^', '**')
        
        # Buscar patrones conocidos
        for pattern in LHopitalAnalyzer.ZERO_OVER_ZERO_PATTERNS:
            if re.search(pattern, normalized):
                return pattern
        
        # Verificar manualmente algunos casos comunes
        if x_value == 0.0:
            if 'sin(x)/x' in normalized or 'sin(x)/x' in normalized.replace('*', ''):
                return 'sin(x)/x'
            if 'sinh(x)/x' in normalized:
                return 'sinh(x)/x'
            if 'tan(x)/x' in normalized:
                return 'tan(x)/x'
        
        return None
    
    @staticmethod
    def calculate_lhopital_limit(expression: str, x_value: float = 0.0) -> Optional[float]:
        """
        Calcula el límite usando L'Hôpital para expresiones conocidas.
        
        Args:
            expression: Expresión matemática
            x_value: Punto donde calcular el límite
            
        Returns:
            Valor del límite o None si no se puede calcular
        """
        normalized = expression.replace(' ', '').replace('^', '**')
        
        # Solo implementamos casos comunes en x=0
        if x_value != 0.0:
            return None
        
        # Casos específicos implementados
        if 'sin(x)/x' in normalized:
            return 1.0
        elif 'sinh(x)/x' in normalized:
            return 1.0
        elif 'tan(x)/x' in normalized:
            return 1.0
        elif '(1-cos(x))/x**2' in normalized or '(1-cos(x))/(x*x)' in normalized:
            return 0.5
        elif '(exp(x)-1)/x' in normalized:
            return 1.0
        elif 'log(1+x)/x' in normalized:
            return 1.0
        
        return None
    
    @staticmethod
    def suggest_lhopital_application(func: Callable[[float], float], 
                                   expression: str, 
                                   x_value: float) -> Tuple[bool, Optional[float], str]:
        """
        Sugiere si aplicar L'Hôpital y proporciona el valor del límite.
        
        Args:
            func: Función a evaluar
            expression: Expresión original como string
            x_value: Punto problemático
            
        Returns:
            Tupla (es_aplicable, valor_limite, mensaje_explicativo)
        """
        # Verificar si la función es indefinida en el punto
        if not LHopitalAnalyzer.is_function_undefined_at_point(func, x_value):
            return False, None, "La función está definida en este punto."
        
        # Detectar patrón compatible
        pattern = LHopitalAnalyzer.detect_lhopital_pattern(expression, x_value)
        if pattern is None:
            return False, None, (
                f"No se detectó un patrón compatible con L'Hôpital en x={x_value}. "
                "La función podría tener una singularidad no removible."
            )
        
        # Calcular límite
        limit_value = LHopitalAnalyzer.calculate_lhopital_limit(expression, x_value)
        if limit_value is None:
            return False, None, (
                f"Se detectó un patrón ({pattern}) pero no se pudo calcular el límite automáticamente."
            )
        
        message = (
            f"Se detectó una singularidad removible en x={x_value}.\n"
            f"La función {expression} tiene la forma indeterminada 0/0.\n"
            f"Usando L'Hôpital, el límite cuando x→{x_value} es {limit_value}.\n"
            f"¿Desea usar este valor para continuar con la integración?"
        )
        
        return True, limit_value, message


def create_lhopital_aware_function(original_func: Callable[[float], float],
                                 expression: str,
                                 critical_points: dict) -> Callable[[float], float]:
    """
    Crea una versión de la función que maneja singularidades con L'Hôpital.
    
    Args:
        original_func: Función original
        expression: Expresión como string
        critical_points: Diccionario {x_value: limit_value} para puntos críticos
        
    Returns:
        Nueva función que maneja los puntos críticos
    """
    def lhopital_aware_function(x: float) -> float:
        # Verificar si x está en los puntos críticos (con tolerancia)
        for critical_x, limit_value in critical_points.items():
            if abs(x - critical_x) < 1e-15:
                return limit_value
        
        # Si no es un punto crítico, evaluar normalmente
        return original_func(x)
    
    # Mantener metadatos
    lhopital_aware_function.__name__ = f"L'Hôpital-aware {getattr(original_func, '__name__', 'function')}"
    lhopital_aware_function._expression = expression
    lhopital_aware_function._critical_points = critical_points
    
    return lhopital_aware_function


def apply_numerical_lhopital(func: Callable[[float], float], 
                           x_target: float, 
                           epsilon: float = 1e-8) -> Optional[float]:
    """
    Aplica L'Hôpital de forma numérica aproximando las derivadas.
    
    Args:
        func: Función a evaluar
        x_target: Punto donde calcular el límite
        epsilon: Paso para diferencias finitas
        
    Returns:
        Valor del límite o None si no se puede calcular
    """
    try:
        # Intentar evaluar límite por la derecha y por la izquierda
        x_right = x_target + epsilon
        x_left = x_target - epsilon
        
        try:
            f_right = func(x_right)
            f_left = func(x_left)
            
            # Si ambos son finitos y similares, usar el promedio
            if math.isfinite(f_right) and math.isfinite(f_left):
                if abs(f_right - f_left) < 1e-6:
                    return (f_right + f_left) / 2.0
        except:
            pass
        
        # Intentar con epsilon más pequeño
        epsilon = epsilon / 10
        x_right = x_target + epsilon
        
        try:
            f_right = func(x_right)
            if math.isfinite(f_right):
                return f_right
        except:
            pass
        
        return None
        
    except Exception:
        return None
