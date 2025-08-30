"""
Implementación de la Regla de Boole para integración numérica.
Método de Newton-Cotes de grado 4.
"""

from typing import Callable, Optional
from ..utils import calculate_step_size, generate_points, validate_integration_parameters
from .base import NewtonCotesMethod, IntegrationPoint, IntegrationResult


class BooleMethod(NewtonCotesMethod):
    """
    Regla de Boole para integración numérica.
    
    Método de Newton-Cotes de grado 4 que usa polinomios de cuarto grado.
    Requiere que n sea múltiplo de 4 para aplicación directa.
    
    Fórmula: I ≈ (2h/45)[7f(x₀) + 32f(x₁) + 12f(x₂) + 32f(x₃) + 14f(x₄) + ...]
    Error: E = -8(b-a)⁷f⁽⁶⁾(ξ)/(945n⁶)
    """
    
    def __init__(self):
        """Inicializa Boole con coeficientes [7, 32, 12, 32, 7]."""
        super().__init__(name="Boole", degree=4, coefficients=[7.0, 32.0, 12.0, 32.0, 7.0])
    
    def integrate(self, func: Callable[[float], float], a: float, b: float, 
                 n: int) -> IntegrationResult:
        """
        Integra usando la regla de Boole.
        
        Nota: n debe ser múltiplo de 4 para Boole
        """
        validate_integration_parameters(a, b, n)
        
        if n % 4 != 0:
            raise ValueError("La regla de Boole requiere que n sea múltiplo de 4")
        
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
            
            # Coeficientes Boole: patrón [7, 32, 12, 32, 14, 32, 12, 32, ...]
            coefficient = self._get_boole_coefficient(i, n)
            
            contribution = (2 * h / 45) * coefficient * fx
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
    
    def _get_boole_coefficient(self, i: int, n: int) -> float:
        """
        Obtiene el coeficiente de Boole para el índice i.
        
        Patrón: 7, 32, 12, 32, 14, 32, 12, 32, 14, ..., 32, 12, 32, 7
        """
        if i == 0 or i == n:  # Extremos
            return 7.0
        elif i % 4 == 0:  # Múltiplos de 4 internos
            return 14.0  # Nota: 7 + 7 = 14 cuando se superponen intervalos
        elif i % 4 == 1 or i % 4 == 3:  # Posiciones 1 y 3 en cada grupo de 4
            return 32.0
        else:  # Posición 2 en cada grupo de 4
            return 12.0
    
    def theoretical_error(self, a: float, b: float, n: int, 
                         derivative_bound: float = None) -> Optional[float]:
        """
        Error teórico: E = -8(b-a)⁷f⁽⁶⁾(ξ)/(945n⁶)
        
        Nota: Requiere cota de la sexta derivada
        """
        if derivative_bound is None:
            return None
        
        if derivative_bound < 0:
            raise ValueError("La cota de la derivada debe ser no negativa")
        
        error_bound = 8 * (abs(b - a) ** 7) * derivative_bound / (945 * n ** 6)
        return error_bound
    
    def get_method_info(self) -> dict:
        """Información detallada sobre el método de Boole."""
        base_info = super().get_method_info() if hasattr(super(), 'get_method_info') else {}
        
        boole_info = {
            'name': self.name,
            'degree': self.degree,
            'type': 'Newton-Cotes cerrada',
            'coefficients': self.coefficients,
            'coefficient_pattern': '[7, 32, 12, 32, 14, 32, 12, 32, ...]',
            'formula': 'I ≈ (2h/45)[7f(x₀) + 32f(x₁) + 12f(x₂) + 32f(x₃) + 7f(x₄)]',
            'error_order': 'O(h⁷)',
            'error_formula': 'E = -8(b-a)⁷f⁽⁶⁾(ξ)/(945n⁶)',
            'points_per_interval': 5,
            'subdivision_requirement': 'n debe ser múltiplo de 4',
            'geometric_interpretation': 'suma de áreas bajo polinomios de grado 4',
            'advantages': [
                'Alta precisión para funciones suaves',
                'Excelente para polinomios de grado ≤ 4',
                'Error de orden muy alto O(h⁷)'
            ],
            'disadvantages': [
                'Requiere muchas evaluaciones de función',
                'Sensible a errores de redondeo',
                'Restrictivo en número de subdivisiones',
                'Inestable para funciones oscilatorias'
            ]
        }
        
        return {**base_info, **boole_info}


# Función de compatibilidad
def regla_boole(func: Callable[[float], float], a: float, b: float, 
               n: int) -> tuple:
    """Función de compatibilidad para la regla de Boole."""
    method = BooleMethod()
    result = method.integrate(func, a, b, n)
    
    history = [(point.index, point.x, point.fx, point.coefficient, point.contribution) 
               for point in result.points]
    
    return result.value, history, result.step_size
