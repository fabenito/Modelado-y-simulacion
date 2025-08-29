import unittest
import math
from simulador_integracion_numerica import (
    _make_safe_func,
    _eval_safe_expression,
    regla_trapezoidal,
    regla_simpson_1_3,
    regla_simpson_3_8,
    regla_boole,
    integracion_adaptativa_simpson
)


class TestSimuladorIntegracionNumerica(unittest.TestCase):

    def test_make_safe_func(self):
        func = _make_safe_func("x**2")
        self.assertAlmostEqual(func(2), 4)
        self.assertAlmostEqual(func(3), 9)

    def test_eval_safe_expression(self):
        self.assertAlmostEqual(_eval_safe_expression("pi"), math.pi)
        self.assertAlmostEqual(_eval_safe_expression("pi/2"), math.pi / 2)
        self.assertAlmostEqual(_eval_safe_expression("sqrt(2)"), math.sqrt(2))

    def test_regla_trapezoidal(self):
        func = lambda x: x**2
        result, history, h = regla_trapezoidal(func, 0, 1, 10)
        self.assertAlmostEqual(result, 1/3, places=4)

    def test_regla_simpson_1_3(self):
        func = lambda x: x**2
        result, history, h = regla_simpson_1_3(func, 0, 1, 10)
        self.assertAlmostEqual(result, 1/3, places=4)

    def test_regla_simpson_3_8(self):
        func = lambda x: x**2
        result, history, h = regla_simpson_3_8(func, 0, 1, 9)
        self.assertAlmostEqual(result, 1/3, places=4)

    def test_regla_boole(self):
        func = lambda x: x**2
        result, history, h = regla_boole(func, 0, 1, 8)
        self.assertAlmostEqual(result, 1/3, places=4)

    def test_integracion_adaptativa_simpson(self):
        func = lambda x: x**2
        result, history = integracion_adaptativa_simpson(func, 0, 1, tol=1e-6)
        self.assertAlmostEqual(result, 1/3, places=6)


if __name__ == "__main__":
    unittest.main()
