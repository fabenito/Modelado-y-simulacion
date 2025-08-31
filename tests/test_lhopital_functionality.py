"""
Tests para la funcionalidad de L'Hôpital en el simulador de integración numérica.
"""

import unittest
import math
import sys
import os

# Agregar el path del proyecto
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_path)

from integracion_numerica.utils import (
    make_safe_function, 
    check_for_singularities,
    LHopitalAnalyzer,
    create_lhopital_aware_function
)


class TestLHopitalFunctionality(unittest.TestCase):
    """Tests para la funcionalidad de L'Hôpital."""
    
    def test_singularity_detection_sin_x_over_x(self):
        """Test detección de singularidad en sin(x)/x."""
        expr = "sin(x)/x"
        
        # Crear función sin L'Hôpital (debería fallar en x=0)
        func_normal = make_safe_function(expr)
        
        # Verificar que es indefinida en x=0
        with self.assertRaises(ValueError):
            func_normal(0.0)
        
        # Verificar detección de singularidades
        has_sing, critical_points, message = check_for_singularities(expr, 0.0, 1.0)
        
        self.assertTrue(has_sing)
        self.assertEqual(len(critical_points), 1)
        self.assertEqual(critical_points[0][0], 0.0)  # x=0
        self.assertAlmostEqual(critical_points[0][1], 1.0, places=10)  # límite=1
    
    def test_lhopital_analyzer_patterns(self):
        """Test detección de patrones por LHopitalAnalyzer."""
        analyzer = LHopitalAnalyzer()
        
        # Test sin(x)/x
        self.assertIsNotNone(analyzer.detect_lhopital_pattern("sin(x)/x", 0.0))
        self.assertAlmostEqual(analyzer.calculate_lhopital_limit("sin(x)/x", 0.0), 1.0)
        
        # Test sinh(x)/x
        self.assertIsNotNone(analyzer.detect_lhopital_pattern("sinh(x)/x", 0.0))
        self.assertAlmostEqual(analyzer.calculate_lhopital_limit("sinh(x)/x", 0.0), 1.0)
        
        # Test expresión normal (no debería detectar patrón)
        self.assertIsNone(analyzer.detect_lhopital_pattern("x**2 + 1", 0.0))
    
    def test_function_undefined_at_point(self):
        """Test verificación de función indefinida."""
        # sin(x)/x indefinida en x=0
        func_sin_over_x = make_safe_function("sin(x)/x")
        self.assertTrue(LHopitalAnalyzer.is_function_undefined_at_point(func_sin_over_x, 0.0))
        
        # x^2 definida en x=0
        func_x_squared = make_safe_function("x**2")
        self.assertFalse(LHopitalAnalyzer.is_function_undefined_at_point(func_x_squared, 0.0))
    
    def test_lhopital_aware_function_creation(self):
        """Test creación de función con L'Hôpital."""
        expr = "sin(x)/x"
        lhopital_points = {0.0: 1.0}
        
        func = make_safe_function(expr, lhopital_points)
        
        # Verificar que funciona en x=0
        self.assertAlmostEqual(func(0.0), 1.0, places=10)
        
        # Verificar que funciona en otros puntos
        self.assertAlmostEqual(func(math.pi/2), 2/math.pi, places=6)
    
    def test_integration_with_lhopital(self):
        """Test integración completa con L'Hôpital."""
        from integracion_numerica.methods import Simpson13Method
        
        expr = "sin(x)/x"
        a, b = 0.0, math.pi
        
        # Sin L'Hôpital - debería fallar
        func_normal = make_safe_function(expr)
        method = Simpson13Method()
        
        with self.assertRaises(ValueError):
            method.integrate(func_normal, a, b, 10)
        
        # Con L'Hôpital - debería funcionar
        lhopital_points = {0.0: 1.0}
        func_lhopital = make_safe_function(expr, lhopital_points)
        
        result = method.integrate(func_lhopital, a, b, 10)
        
        # Verificar que obtuvimos un resultado válido
        self.assertIsInstance(result.value, float)
        self.assertTrue(math.isfinite(result.value))
        self.assertGreater(result.value, 0)  # sin(x)/x integrada de 0 a π debería ser positiva
    
    def test_multiple_singularities(self):
        """Test manejo de múltiples singularidades (caso teórico)."""
        # Nota: Este es un caso sintético para testing
        expr = "sin(x)/x"  # Solo singularidad en x=0
        
        has_sing, critical_points, message = check_for_singularities(expr, 0.0, 1.0)
        
        self.assertTrue(has_sing)
        self.assertEqual(len(critical_points), 1)
        self.assertIn("sin(x)/x", message.lower() or message.lower())
    
    def test_no_singularities(self):
        """Test función sin singularidades."""
        expr = "x**2 + 1"
        
        has_sing, critical_points, message = check_for_singularities(expr, 0.0, 1.0)
        
        self.assertFalse(has_sing)
        self.assertEqual(len(critical_points), 0)
    
    def test_lhopital_metadata_preservation(self):
        """Test que los metadatos se preservan en funciones con L'Hôpital."""
        expr = "sin(x)/x"
        lhopital_points = {0.0: 1.0}
        
        func = make_safe_function(expr, lhopital_points)
        
        self.assertEqual(func._expression, expr)
        self.assertEqual(func._lhopital_points, lhopital_points)
        self.assertIn("sin(x)/x", func.__name__)


if __name__ == '__main__':
    # Ejecutar tests
    unittest.main(verbosity=2)
