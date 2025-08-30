"""
Implementación de la Regla Trapezoidal para integración numérica.
Método de Newton-Cotes de grado 1.
"""

from typing import Callable, List, Optional
from ..utils import calculate_step_size, generate_points, validate_integration_parameters
from .base import NewtonCotesMethod, IntegrationPoint, IntegrationResult


class TrapezoidalMethod(NewtonCotesMethod):
    """
    Regla Trapezoidal para integración numérica.
    
    Método de Newton-Cotes de grado 1 que aproxima la función con
    líneas rectas entre puntos adyacentes y calcula el área bajo estos
    segmentos lineales.
    
    Fórmula: I ≈ (h/2)[f(a) + 2f(x₁) + 2f(x₂) + ... + 2f(xₙ₋₁) + f(b)]
    Error: E = -(b-a)³f''(ξ)/(12n²)
    """
    
    def __init__(self):
        """Inicializa el método trapezoidal con coeficientes [1, 1]."""
        super().__init__(name="Trapezoidal", degree=1, coefficients=[1.0, 1.0])
    
    def integrate(self, func: Callable[[float], float], a: float, b: float, 
                 n: int) -> IntegrationResult:
        """
        Integra usando la regla trapezoidal.
        
        Args:
            func: Función a integrar f(x)
            a: Límite inferior  
            b: Límite superior
            n: Número de subdivisiones
            
        Returns:
            IntegrationResult con detalles completos del cálculo
        """
        validate_integration_parameters(a, b, n)
        
        h = calculate_step_size(a, b, n)
        x_points = generate_points(a, b, n)
        
        points = []
        integral = 0.0
        function_evaluations = 0
        
        for i, x in enumerate(x_points):
            # Evaluar función
            try:
                fx = func(x)
                function_evaluations += 1
            except Exception as e:
                raise ValueError(f"Error evaluando función en x={x}: {str(e)}")
            
            # Determinar coeficiente
            if i == 0 or i == n:  # Extremos
                coefficient = 1.0
            else:  # Puntos internos
                coefficient = 2.0
            
            # Contribución al resultado final
            contribution = (h / 2) * coefficient * fx
            integral += contribution
            
            # Guardar información del punto
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
        """
        Error teórico: E = -(b-a)³f''(ξ)/(12n²)
        """
        if derivative_bound is None:
            return None
        
        if derivative_bound < 0:
            raise ValueError("La cota de la derivada debe ser no negativa")
        
        error_bound = (abs(b - a) ** 3) * derivative_bound / (12 * n * n)
        return error_bound


# Función de compatibilidad
def regla_trapezoidal(func: Callable[[float], float], a: float, b: float, 
                     n: int) -> tuple:
    """Función de compatibilidad con interfaz original."""
    method = TrapezoidalMethod()
    result = method.integrate(func, a, b, n)
    
    history = [(point.index, point.x, point.fx, point.coefficient, point.contribution) 
               for point in result.points]
    
    return result.value, history, result.step_size
