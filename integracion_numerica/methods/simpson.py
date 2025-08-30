"""
Implementaciones de las Reglas de Simpson para integración numérica.
Métodos de Newton-Cotes de grados 2 y 3.
"""

from typing import Callable, Optional
from ..utils import calculate_step_size, generate_points, validate_integration_parameters
from .base import NewtonCotesMethod, IntegrationPoint, IntegrationResult


class Simpson13Method(NewtonCotesMethod):
    """
    Regla de Simpson 1/3 para integración numérica.
    
    Método de Newton-Cotes de grado 2 que aproxima la función con
    parábolas en cada par de subintervalos.
    
    Fórmula: I ≈ (h/3)[f(a) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + f(b)]
    Error: E = -(b-a)⁵f⁽⁴⁾(ξ)/(180n⁴)
    """
    
    def __init__(self):
        """Inicializa Simpson 1/3 con coeficientes [1, 4, 1]."""
        super().__init__(name="Simpson 1/3", degree=2, coefficients=[1.0, 4.0, 1.0])
    
    def integrate(self, func: Callable[[float], float], a: float, b: float, 
                 n: int) -> IntegrationResult:
        """
        Integra usando la regla de Simpson 1/3.
        
        Nota: n debe ser par para Simpson 1/3
        """
        validate_integration_parameters(a, b, n)
        
        if n % 2 != 0:
            raise ValueError("Simpson 1/3 requiere número par de subdivisiones")
        
        h = calculate_step_size(a, b, n)
        x_points = generate_points(a, b, n)
        
        points = []
        integral = 0.0
        function_evaluations = 0
        
        for i, x in enumerate(x_points):
            try:
                fx = func(x)
                function_evaluations += 1
            except Exception as e:
                raise ValueError(f"Error evaluando función en x={x}: {str(e)}")
            
            # Coeficientes Simpson 1/3: 1, 4, 2, 4, 2, ..., 4, 1
            if i == 0 or i == n:  # Extremos
                coefficient = 1.0
            elif i % 2 == 1:  # Índices impares
                coefficient = 4.0
            else:  # Índices pares internos
                coefficient = 2.0
            
            contribution = (h / 3) * coefficient * fx
            integral += contribution
            
            point = IntegrationPoint(
                index=i,
                x=x,
                fx=fx,
                coefficient=coefficient,
                contribution=contribution
            )
            points.append(point)
        
        return IntegrationResult(
            value=integral,
            error_estimate=None,
            points=points,
            step_size=h,
            method_name=self.name,
            function_evaluations=function_evaluations
        )
    
    def theoretical_error(self, a: float, b: float, n: int, 
                         derivative_bound: float = None) -> Optional[float]:
        """Error teórico: E = -(b-a)⁵f⁽⁴⁾(ξ)/(180n⁴)"""
        if derivative_bound is None:
            return None
        
        if derivative_bound < 0:
            raise ValueError("La cota de la derivada debe ser no negativa")
        
        error_bound = (abs(b - a) ** 5) * derivative_bound / (180 * n ** 4)
        return error_bound


class Simpson38Method(NewtonCotesMethod):
    """
    Regla de Simpson 3/8 para integración numérica.
    
    Método de Newton-Cotes de grado 3 que usa polinomios cúbicos.
    Requiere que n sea múltiplo de 3.
    
    Fórmula: I ≈ (3h/8)[f(x₀) + 3f(x₁) + 3f(x₂) + 2f(x₃) + 3f(x₄) + ...]
    Error: E = -3(b-a)⁵f⁽⁴⁾(ξ)/(80n⁴)
    """
    
    def __init__(self):
        """Inicializa Simpson 3/8 con coeficientes [1, 3, 3, 1]."""
        super().__init__(name="Simpson 3/8", degree=3, coefficients=[1.0, 3.0, 3.0, 1.0])
    
    def integrate(self, func: Callable[[float], float], a: float, b: float, 
                 n: int) -> IntegrationResult:
        """
        Integra usando la regla de Simpson 3/8.
        
        Nota: n debe ser múltiplo de 3 para Simpson 3/8
        """
        validate_integration_parameters(a, b, n)
        
        if n % 3 != 0:
            raise ValueError("Simpson 3/8 requiere que n sea múltiplo de 3")
        
        h = calculate_step_size(a, b, n)
        x_points = generate_points(a, b, n)
        
        points = []
        integral = 0.0
        function_evaluations = 0
        
        for i, x in enumerate(x_points):
            try:
                fx = func(x)
                function_evaluations += 1
            except Exception as e:
                raise ValueError(f"Error evaluando función en x={x}: {str(e)}")
            
            # Coeficientes Simpson 3/8: patrón [1, 3, 3, 2, 3, 3, 2, ..., 3, 3, 1]
            if i == 0 or i == n:  # Extremos
                coefficient = 1.0
            elif i % 3 == 0:  # Múltiplos de 3 internos
                coefficient = 2.0
            else:  # Resto de posiciones
                coefficient = 3.0
            
            contribution = (3 * h / 8) * coefficient * fx
            integral += contribution
            
            point = IntegrationPoint(
                index=i,
                x=x,
                fx=fx,
                coefficient=coefficient,
                contribution=contribution
            )
            points.append(point)
        
        return IntegrationResult(
            value=integral,
            error_estimate=None,
            points=points,
            step_size=h,
            method_name=self.name,
            function_evaluations=function_evaluations
        )
    
    def theoretical_error(self, a: float, b: float, n: int, 
                         derivative_bound: float = None) -> Optional[float]:
        """Error teórico: E = -3(b-a)⁵f⁽⁴⁾(ξ)/(80n⁴)"""
        if derivative_bound is None:
            return None
        
        if derivative_bound < 0:
            raise ValueError("La cota de la derivada debe ser no negativa")
        
        error_bound = 3 * (abs(b - a) ** 5) * derivative_bound / (80 * n ** 4)
        return error_bound


# Funciones de compatibilidad
def regla_simpson_13(func: Callable[[float], float], a: float, b: float, 
                    n: int) -> tuple:
    """Función de compatibilidad para Simpson 1/3."""
    method = Simpson13Method()
    result = method.integrate(func, a, b, n)
    
    history = [(point.index, point.x, point.fx, point.coefficient, point.contribution) 
               for point in result.points]
    
    return result.value, history, result.step_size


def regla_simpson_38(func: Callable[[float], float], a: float, b: float, 
                    n: int) -> tuple:
    """Función de compatibilidad para Simpson 3/8."""
    method = Simpson38Method()
    result = method.integrate(func, a, b, n)
    
    history = [(point.index, point.x, point.fx, point.coefficient, point.contribution) 
               for point in result.points]
    
    return result.value, history, result.step_size
