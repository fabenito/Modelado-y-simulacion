"""
Implementación de la Regla del Rectángulo (Punto Medio) para integración numérica.
Método de Newton-Cotes de grado 0.
"""

from typing import Callable, List, Optional
from ..utils import calculate_step_size, generate_points, validate_integration_parameters
from .base import NewtonCotesMethod, IntegrationPoint, IntegrationResult


class RectangleMethod(NewtonCotesMethod):
    """
    Regla del Rectángulo/Punto Medio para integración numérica.
    
    Este es el método más simple de Newton-Cotes (grado 0), que aproxima
    la integral usando el valor de la función en el punto medio de cada
    subintervalo multiplicado por el ancho del intervalo.
    
    Fórmula: I ≈ h * Σ f((x_i + x_{i+1})/2)
    Error: E = (b-a)³f''(ξ)/(24n²)
    """
    
    def __init__(self):
        """Inicializa el método del rectángulo con coeficiente [1]."""
        super().__init__(name="Rectángulo/Punto Medio", degree=0, coefficients=[1.0])
    
    def integrate(self, func: Callable[[float], float], a: float, b: float, 
                 n: int) -> IntegrationResult:
        """
        Integra usando la regla del rectángulo con evaluación en punto medio.
        
        Args:
            func: Función a integrar f(x)
            a: Límite inferior  
            b: Límite superior
            n: Número de subdivisiones (rectángulos)
            
        Returns:
            IntegrationResult con detalles completos del cálculo
            
        Raises:
            ValueError: Si los parámetros no son válidos
        """
        # Validar parámetros
        validate_integration_parameters(a, b, n)
        
        # Calcular tamaño de paso
        h = calculate_step_size(a, b, n)
        
        # Lista para almacenar información de cada punto
        points = []
        integral = 0.0
        function_evaluations = 0
        
        # Calcular integral usando punto medio de cada subintervalo
        for i in range(n):
            # Límites del subintervalo i
            x_left = a + i * h
            x_right = a + (i + 1) * h
            
            # Punto medio del subintervalo
            x_mid = (x_left + x_right) / 2.0
            
            # Evaluar función en el punto medio
            try:
                fx_mid = func(x_mid)
                function_evaluations += 1
            except Exception as e:
                raise ValueError(f"Error evaluando función en x={x_mid}: {str(e)}")
            
            # Contribución de este rectángulo
            coefficient = 1.0  # Coeficiente siempre es 1 para rectángulos
            contribution = h * fx_mid
            integral += contribution
            
            # Guardar información del punto
            point = IntegrationPoint(
                index=i,
                x=x_mid,
                fx=fx_mid,
                coefficient=coefficient,
                contribution=contribution
            )
            points.append(point)
        
        # Crear resultado
        result = IntegrationResult(
            value=integral,
            error_estimate=None,  # Error teórico requiere información de derivada
            points=points,
            step_size=h,
            method_name=self.name,
            function_evaluations=function_evaluations
        )
        
        return result
    
    def theoretical_error(self, a: float, b: float, n: int, 
                         derivative_bound: float = None) -> Optional[float]:
        """
        Calcula el error teórico de la regla del rectángulo.
        
        Error teórico: E = (b-a)³f''(ξ)/(24n²)
        
        Args:
            a: Límite inferior
            b: Límite superior  
            n: Número de subdivisiones
            derivative_bound: Cota superior de |f''(x)| en [a,b]
            
        Returns:
            Estimación del error absoluto o None si no se puede calcular
        """
        if derivative_bound is None:
            return None
        
        if derivative_bound < 0:
            raise ValueError("La cota de la derivada debe ser no negativa")
        
        # Fórmula del error: E = (b-a)³M₂/(24n²)
        # donde M₂ es la cota de |f''(x)|
        error_bound = (abs(b - a) ** 3) * derivative_bound / (24 * n * n)
        
        return error_bound
    
    def get_method_info(self) -> dict:
        """
        Retorna información detallada sobre el método.
        
        Returns:
            Diccionario con información técnica del método
        """
        return {
            'name': self.name,
            'degree': self.degree,
            'type': 'Newton-Cotes cerrada',
            'coefficients': self.coefficients,
            'formula': 'I ≈ h * Σ f((x_i + x_{i+1})/2)',
            'error_order': 'O(h³)',
            'error_formula': 'E = (b-a)³f\'\'(ξ)/(24n²)',
            'points_per_interval': 1,
            'evaluation_strategy': 'punto medio',
            'geometric_interpretation': 'suma de áreas de rectángulos',
            'advantages': [
                'Muy simple de implementar',
                'Computacionalmente eficiente',
                'Buena para funciones suaves',
                'Método base para cuadratura adaptativa'
            ],
            'disadvantages': [
                'Convergencia lenta (O(h²))',
                'No aprovecha derivadas',
                'Sensible a discontinuidades'
            ]
        }
    
    def __str__(self) -> str:
        """Representación string del método."""
        return f"Regla del Rectángulo (Punto Medio) - Grado 0"


# Función de conveniencia para compatibilidad con código existente
def regla_rectangulo(func: Callable[[float], float], a: float, b: float, 
                    n: int) -> tuple:
    """
    Función de conveniencia para la regla del rectángulo.
    Mantiene compatibilidad con la interfaz original.
    
    Args:
        func: Función a integrar
        a: Límite inferior
        b: Límite superior  
        n: Número de subdivisiones
        
    Returns:
        Tupla (resultado, historia, h) compatible con interfaz original
    """
    method = RectangleMethod()
    result = method.integrate(func, a, b, n)
    
    # Convertir resultado a formato de historia compatible
    history = [(point.index, point.x, point.fx, point.coefficient, point.contribution) 
               for point in result.points]
    
    return result.value, history, result.step_size
