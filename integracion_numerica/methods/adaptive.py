"""
Implementación del método adaptativo de Simpson para integración numérica.
Ajusta automáticamente el tamaño de paso basándose en estimaciones de error.
"""

from typing import Callable, List, Tuple, Optional
from ..utils import validate_integration_parameters
from .base import AdaptiveMethod, IntegrationPoint, IntegrationResult
from .simpson import Simpson13Method


class AdaptiveSimpsonMethod(AdaptiveMethod):
    """
    Método adaptativo basado en la regla de Simpson 1/3.
    
    Ajusta automáticamente el tamaño de paso dividiendo recursivamente
    los intervalos hasta alcanzar la precisión deseada.
    
    Estimación de error: |S_h - S_{2h}|/15 ≤ ε
    donde S_h es Simpson con paso h y S_{2h} es Simpson con paso 2h
    """
    
    def __init__(self, tolerance: float = 1e-6, max_depth: int = 15):
        """
        Inicializa el método adaptativo de Simpson.
        
        Args:
            tolerance: Tolerancia de error objetivo
            max_depth: Máxima profundidad de recursión
        """
        simpson_method = Simpson13Method()
        super().__init__(
            name="Simpson Adaptativo",
            base_method=simpson_method,
            tolerance=tolerance,
            max_depth=max_depth
        )
    
    def integrate(self, func: Callable[[float], float], a: float, b: float, 
                 n: int = None) -> IntegrationResult:
        """
        Integra usando el método adaptativo de Simpson.
        
        Args:
            func: Función a integrar
            a: Límite inferior
            b: Límite superior
            n: No usado (método adaptativo ajusta automáticamente)
            
        Returns:
            IntegrationResult con detalles del proceso adaptativo
        """
        validate_integration_parameters(a, b, 2)  # Mínimo 2 subdivisiones
        
        # Listas para almacenar información del proceso
        all_points = []
        function_evaluations = 0
        
        def simpson_basic(x0: float, x2: float) -> Tuple[float, int]:
            """Aplica Simpson básico en un intervalo [x0, x2]."""
            nonlocal function_evaluations
            
            x1 = (x0 + x2) / 2
            h = (x2 - x0) / 2
            
            try:
                f0 = func(x0)
                f1 = func(x1) 
                f2 = func(x2)
                function_evaluations += 3
            except Exception as e:
                raise ValueError(f"Error evaluando función: {str(e)}")
            
            result = (h / 3) * (f0 + 4 * f1 + f2)
            return result, 3
        
        def adaptive_simpson_recursive(x0: float, x2: float, tol: float, 
                                     whole_interval: float, depth: int) -> float:
            """
            Integración recursiva adaptativa de Simpson.
            
            Args:
                x0, x2: Extremos del intervalo actual
                tol: Tolerancia para este intervalo
                whole_interval: Simpson para todo el intervalo [x0, x2]
                depth: Profundidad actual de recursión
                
            Returns:
                Integral adaptativa en el intervalo
            """
            nonlocal all_points, function_evaluations
            
            if depth > self.max_depth:
                # Si se alcanza máxima profundidad, usar resultado actual
                return whole_interval
            
            # Punto medio del intervalo
            x1 = (x0 + x2) / 2
            
            # Simpson en mitades izquierda y derecha
            left_simpson, _ = simpson_basic(x0, x1)
            right_simpson, _ = simpson_basic(x1, x2)
            sum_halves = left_simpson + right_simpson
            
            # Estimación de error usando regla de Richardson
            error_estimate = abs(sum_halves - whole_interval) / 15
            
            # Guardar información del punto medio
            try:
                fx1 = func(x1)
                point = IntegrationPoint(
                    index=len(all_points),
                    x=x1,
                    fx=fx1,
                    coefficient=4.0,  # Coeficiente típico de Simpson
                    contribution=error_estimate  # Usar error como contribución
                )
                all_points.append(point)
            except:
                pass  # En caso de error, continuar sin guardar el punto
            
            if error_estimate <= tol:
                # Tolerancia satisfecha, aceptar suma de mitades
                return sum_halves
            else:
                # Dividir intervalo y procesar recursivamente
                tol_half = tol / 2  # Distribuir tolerancia
                left_result = adaptive_simpson_recursive(
                    x0, x1, tol_half, left_simpson, depth + 1
                )
                right_result = adaptive_simpson_recursive(
                    x1, x2, tol_half, right_simpson, depth + 1
                )
                return left_result + right_result
        
        # Aplicar Simpson inicial en todo el intervalo
        initial_simpson, _ = simpson_basic(a, b)
        
        # Comenzar recursión adaptativa
        final_result = adaptive_simpson_recursive(
            a, b, self.tolerance, initial_simpson, 1
        )
        
        # Crear resultado
        step_size = (b - a) / max(len(all_points), 2)  # Paso estimado
        
        return IntegrationResult(
            value=final_result,
            error_estimate=self.tolerance,  # Tolerancia alcanzada
            points=all_points,
            step_size=step_size,
            method_name=self.name,
            function_evaluations=function_evaluations
        )
    
    def get_method_info(self) -> dict:
        """Información detallada sobre el método adaptativo."""
        return {
            'name': self.name,
            'type': 'Adaptativo basado en Simpson',
            'base_method': 'Simpson 1/3',
            'tolerance': self.tolerance,
            'max_depth': self.max_depth,
            'error_estimation': '|S_h - S_{2h}|/15',
            'strategy': 'División recursiva de intervalos',
            'advantages': [
                'Ajuste automático de precisión',
                'Eficiente para funciones variables',
                'Control de error robusto',
                'Concentra esfuerzo donde se necesita'
            ],
            'disadvantages': [
                'Difícil de predecir costo computacional',
                'Puede ser ineficiente para funciones suaves',
                'Complejidad de implementación mayor'
            ],
            'best_for': [
                'Funciones con comportamiento irregular',
                'Integrales con singularidades suaves',
                'Cuando se requiere precisión garantizada'
            ]
        }


# Función de compatibilidad
def integracion_adaptativa_simpson(func: Callable[[float], float], a: float, b: float, 
                                 tol: float = 1e-6) -> Tuple[float, List]:
    """
    Función de compatibilidad para integración adaptativa.
    
    Returns:
        Tupla (resultado, historia) compatible con interfaz original
    """
    method = AdaptiveSimpsonMethod(tolerance=tol)
    result = method.integrate(func, a, b)
    
    # Convertir a formato de historia compatible
    # Para método adaptativo, la historia es diferente
    history = []
    for i, point in enumerate(result.points):
        # Formato: (intervalo_info, punto_medio, evaluación, error_estimado)
        history_entry = {
            'depth': i,
            'x': point.x,
            'fx': point.fx,
            'error_estimate': point.contribution,
            'interval_contribution': point.coefficient
        }
        history.append(history_entry)
    
    return result.value, history
