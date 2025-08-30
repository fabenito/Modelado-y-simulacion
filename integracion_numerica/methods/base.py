"""
Clases base para métodos de integración numérica.
Proporciona estructura común y tipos de datos para todos los métodos.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Any
import math


@dataclass
class IntegrationPoint:
    """Representa un punto en el proceso de integración numérica."""
    index: int
    x: float
    fx: float
    coefficient: float
    contribution: float
    
    def __post_init__(self):
        """Validación automática después de inicialización."""
        if not math.isfinite(self.x):
            raise ValueError(f"Coordenada x no finita: {self.x}")
        if not math.isfinite(self.fx):
            raise ValueError(f"Valor de función no finito en x={self.x}: {self.fx}")


@dataclass 
class IntegrationResult:
    """Resultado completo de un método de integración numérica."""
    value: float
    error_estimate: Optional[float]
    points: List[IntegrationPoint]
    step_size: float
    method_name: str
    function_evaluations: int
    
    def __post_init__(self):
        """Validación del resultado."""
        if not math.isfinite(self.value):
            raise ValueError(f"Resultado de integración no finito: {self.value}")
        if self.step_size <= 0:
            raise ValueError(f"Tamaño de paso debe ser positivo: {self.step_size}")
        if self.function_evaluations <= 0:
            raise ValueError(f"Número de evaluaciones debe ser positivo: {self.function_evaluations}")


class IntegrationMethod(ABC):
    """
    Clase base abstracta para todos los métodos de integración numérica.
    
    Define la interface común que deben implementar todos los métodos
    de integración de Newton-Cotes y adaptativos.
    """
    
    def __init__(self, name: str, degree: int):
        """
        Inicializa el método de integración.
        
        Args:
            name: Nombre del método (ej. "Trapezoidal", "Simpson 1/3")
            degree: Grado del polinomio interpolante (0=Rectangle, 1=Trapezoid, etc.)
        """
        self.name = name
        self.degree = degree
    
    @abstractmethod
    def integrate(self, func: Callable[[float], float], a: float, b: float, 
                 n: int) -> IntegrationResult:
        """
        Método principal de integración que deben implementar las subclases.
        
        Args:
            func: Función a integrar f(x)
            a: Límite inferior de integración
            b: Límite superior de integración  
            n: Número de subdivisiones del intervalo
            
        Returns:
            IntegrationResult con todos los detalles del cálculo
        """
        pass
    
    @abstractmethod
    def theoretical_error(self, a: float, b: float, n: int, 
                         derivative_bound: float = None) -> Optional[float]:
        """
        Calcula el error teórico del método si es posible.
        
        Args:
            a: Límite inferior
            b: Límite superior
            n: Número de subdivisiones
            derivative_bound: Cota superior de la derivada relevante
            
        Returns:
            Estimación del error teórico o None si no se puede calcular
        """
        pass
    
    def __str__(self) -> str:
        """Representación string del método."""
        return f"{self.name} (Grado {self.degree})"
    
    def __repr__(self) -> str:
        """Representación técnica del método."""
        return f"IntegrationMethod(name='{self.name}', degree={self.degree})"


class NewtonCotesMethod(IntegrationMethod):
    """
    Clase base para métodos de Newton-Cotes de fórmula cerrada.
    
    Los métodos de Newton-Cotes usan puntos equidistantes y coeficientes
    predeterminados basados en interpolación polinomial.
    """
    
    def __init__(self, name: str, degree: int, coefficients: List[float]):
        """
        Inicializa un método de Newton-Cotes.
        
        Args:
            name: Nombre del método
            degree: Grado del polinomio interpolante
            coefficients: Coeficientes de Newton-Cotes para el método
        """
        super().__init__(name, degree)
        self.coefficients = coefficients
        self._validate_coefficients()
    
    def _validate_coefficients(self):
        """Valida que los coeficientes sean correctos."""
        if not self.coefficients:
            raise ValueError("Los coeficientes no pueden estar vacíos")
        if len(self.coefficients) != self.degree + 1:
            raise ValueError(f"Se esperan {self.degree + 1} coeficientes para grado {self.degree}")
        if not all(math.isfinite(c) for c in self.coefficients):
            raise ValueError("Todos los coeficientes deben ser números finitos")
    
    def get_coefficients_info(self) -> dict:
        """
        Retorna información sobre los coeficientes del método.
        
        Returns:
            Diccionario con información de coeficientes
        """
        return {
            'coefficients': self.coefficients.copy(),
            'sum': sum(self.coefficients),
            'degree': self.degree,
            'symmetry': self.coefficients == self.coefficients[::-1]
        }


class AdaptiveMethod(IntegrationMethod):
    """
    Clase base para métodos adaptativos de integración.
    
    Los métodos adaptativos ajustan automáticamente el tamaño de paso
    basándose en estimaciones de error local.
    """
    
    def __init__(self, name: str, base_method: IntegrationMethod, 
                 tolerance: float = 1e-6, max_depth: int = 15):
        """
        Inicializa un método adaptativo.
        
        Args:
            name: Nombre del método adaptativo
            base_method: Método base para las estimaciones
            tolerance: Tolerancia de error objetivo
            max_depth: Máxima profundidad de recursión
        """
        super().__init__(name, base_method.degree)
        self.base_method = base_method
        self.tolerance = tolerance
        self.max_depth = max_depth
        self._validate_parameters()
    
    def _validate_parameters(self):
        """Valida los parámetros del método adaptativo."""
        if self.tolerance <= 0:
            raise ValueError(f"Tolerancia debe ser positiva: {self.tolerance}")
        if self.max_depth <= 0:
            raise ValueError(f"Profundidad máxima debe ser positiva: {self.max_depth}")
        if self.max_depth > 20:
            raise ValueError(f"Profundidad máxima muy grande: {self.max_depth}")
    
    def theoretical_error(self, a: float, b: float, n: int, 
                         derivative_bound: float = None) -> Optional[float]:
        """Los métodos adaptativos usan estimación de error, no error teórico."""
        return None  # Los adaptativos no tienen error teórico simple
