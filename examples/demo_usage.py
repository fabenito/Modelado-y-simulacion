#!/usr/bin/env python3
"""
Ejemplos de uso del simulador de integración numérica refactorizado.
Demuestra cómo usar los diferentes métodos programáticamente.
"""

import sys
import os
import math

# Agregar path del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integracion_numerica.methods import (
    get_all_methods, compare_methods,
    RectangleMethod, TrapezoidalMethod, Simpson13Method, Simpson38Method, 
    BooleMethod, AdaptiveSimpsonMethod
)
from integracion_numerica.utils import make_safe_function, format_result


def ejemplo_basico():
    """Ejemplo básico usando diferentes métodos."""
    print("="*60)
    print("EJEMPLO BÁSICO: Integración de x² en [0, 1]")
    print("="*60)
    
    # Definir función y parámetros
    func = make_safe_function("x**2")
    a, b, n = 0.0, 1.0, 10
    valor_exacto = 1/3  # ∫₀¹ x² dx = 1/3
    
    print(f"Función: f(x) = x²")
    print(f"Intervalo: [{a}, {b}]")
    print(f"Subdivisiones: {n}")
    print(f"Valor exacto: {format_result(valor_exacto)}")
    print()
    
    # Probar diferentes métodos
    methods = get_all_methods()
    
    for name, method in methods.items():
        try:
            if name == 'adaptativo':
                result = method.integrate(func, a, b)
            else:
                # Ajustar n según requerimientos
                n_adj = n
                if name == 'simpson_13' and n % 2 != 0:
                    n_adj = n + 1
                elif name == 'simpson_38' and n % 3 != 0:
                    n_adj = ((n // 3) + 1) * 3
                elif name == 'boole' and n % 4 != 0:
                    n_adj = ((n // 4) + 1) * 4
                
                result = method.integrate(func, a, b, n_adj)
            
            error = abs(result.value - valor_exacto)
            error_rel = error / valor_exacto * 100
            
            print(f"{method.name:20} | "
                  f"Resultado: {format_result(result.value, 8)} | "
                  f"Error: {format_result(error, 6)} ({error_rel:.2f}%) | "
                  f"Eval: {result.function_evaluations:3d}")
                  
        except Exception as e:
            print(f"{name:20} | Error: {str(e)}")
    
    print()


def ejemplo_funciones_trigonometricas():
    """Ejemplo con funciones trigonométricas."""
    print("="*60)
    print("EJEMPLO TRIGONOMÉTRICO: Integración de sin(x) en [0, π]")
    print("="*60)
    
    func = make_safe_function("sin(x)")
    a, b, n = 0.0, math.pi, 12
    valor_exacto = 2.0  # ∫₀^π sin(x) dx = 2
    
    print(f"Función: f(x) = sin(x)")
    print(f"Intervalo: [0, π] ≈ [0, {b:.6f}]")
    print(f"Valor exacto: {valor_exacto}")
    print()
    
    # Método adaptativo con diferentes tolerancias
    tolerancias = [1e-3, 1e-6, 1e-9]
    
    for tol in tolerancias:
        method = AdaptiveSimpsonMethod(tolerance=tol)
        result = method.integrate(func, a, b)
        
        error = abs(result.value - valor_exacto)
        print(f"Adaptativo (tol={tol:1.0e}) | "
              f"Resultado: {format_result(result.value, 10)} | "
              f"Error: {format_result(error, 8)} | "
              f"Evaluaciones: {result.function_evaluations}")


def ejemplo_comparacion_precision():
    """Compara la precisión de diferentes métodos."""
    print("="*60)
    print("COMPARACIÓN DE PRECISIÓN: f(x) = exp(-x²) en [0, 2]")
    print("="*60)
    
    func = make_safe_function("exp(-x**2)")
    a, b = 0.0, 2.0
    
    # Valores de n para probar
    n_values = [4, 8, 16, 32]
    
    print("n\\Método     Rectangle    Trapezoidal   Simpson 1/3   Simpson 3/8   Boole")
    print("-" * 75)
    
    for n in n_values:
        resultados = []
        
        # Rectangle
        try:
            rect = RectangleMethod()
            res = rect.integrate(func, a, b, n)
            resultados.append(format_result(res.value, 6))
        except:
            resultados.append("Error")
        
        # Trapezoidal
        try:
            trap = TrapezoidalMethod()
            res = trap.integrate(func, a, b, n)
            resultados.append(format_result(res.value, 6))
        except:
            resultados.append("Error")
        
        # Simpson 1/3
        try:
            n_adj = n if n % 2 == 0 else n + 1
            simp13 = Simpson13Method()
            res = simp13.integrate(func, a, b, n_adj)
            resultados.append(format_result(res.value, 6))
        except:
            resultados.append("Error")
        
        # Simpson 3/8
        try:
            n_adj = n if n % 3 == 0 else ((n // 3) + 1) * 3
            simp38 = Simpson38Method()
            res = simp38.integrate(func, a, b, n_adj)
            resultados.append(format_result(res.value, 6))
        except:
            resultados.append("Error")
        
        # Boole
        try:
            n_adj = n if n % 4 == 0 else ((n // 4) + 1) * 4
            boole = BooleMethod()
            res = boole.integrate(func, a, b, n_adj)
            resultados.append(format_result(res.value, 6))
        except:
            resultados.append("Error")
        
        print(f"{n:2d}           {resultados[0]:>10} {resultados[1]:>12} {resultados[2]:>12} {resultados[3]:>12} {resultados[4]:>8}")


def ejemplo_uso_individual():
    """Muestra cómo usar métodos individuales."""
    print("="*60)
    print("USO INDIVIDUAL DE MÉTODOS")
    print("="*60)
    
    # Ejemplo con método específico
    func = make_safe_function("1/(1 + x**2)")
    a, b, n = 0.0, 1.0, 10
    
    print("Integrando f(x) = 1/(1+x²) en [0,1] (≈ π/4)")
    print(f"Valor teórico: π/4 ≈ {math.pi/4:.10f}")
    print()
    
    # Usar método Rectangle específico
    rect_method = RectangleMethod()
    result = rect_method.integrate(func, a, b, n)
    
    print("Método del Rectángulo:")
    print(f"  Resultado: {format_result(result.value, 10)}")
    print(f"  Evaluaciones: {result.function_evaluations}")
    print(f"  Tamaño de paso: {format_result(result.step_size, 6)}")
    print()
    
    # Mostrar detalles de algunos puntos
    print("Primeros 5 puntos de evaluación:")
    for i, point in enumerate(result.points[:5]):
        print(f"  Punto {point.index}: x={format_result(point.x, 6)}, "
              f"f(x)={format_result(point.fx, 6)}, "
              f"contribución={format_result(point.contribution, 8)}")
    
    if len(result.points) > 5:
        print(f"  ... y {len(result.points) - 5} puntos más")


def main():
    """Ejecuta todos los ejemplos."""
    print("SIMULADOR DE INTEGRACIÓN NUMÉRICA - EJEMPLOS DE USO")
    print("Versión Refactorizada 2.0.0")
    print()
    
    try:
        ejemplo_basico()
        ejemplo_funciones_trigonometricas()
        ejemplo_comparacion_precision()
        ejemplo_uso_individual()
        
        print("="*60)
        print("EJEMPLOS COMPLETADOS EXITOSAMENTE")
        print("Para usar la interfaz gráfica, ejecutar:")
        print("  python simulador_integracion_refactorizado.py")
        print("="*60)
        
    except Exception as e:
        print(f"Error ejecutando ejemplos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
