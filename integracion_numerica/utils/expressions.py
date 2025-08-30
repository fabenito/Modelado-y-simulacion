"""
Funciones utilitarias para el simulador de integración numérica.
Incluye parsing de expresiones matemáticas y validación de funciones.
"""

import ast
import math
import operator
from typing import Callable, Any


# Operadores seguros para el parser AST
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Funciones matemáticas seguras
SAFE_FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'asin': math.asin,
    'acos': math.acos,
    'atan': math.atan,
    'sinh': math.sinh,
    'cosh': math.cosh,
    'tanh': math.tanh,
    'exp': math.exp,
    'log': math.log,
    'log10': math.log10,
    'sqrt': math.sqrt,
    'abs': abs,
    'ceil': math.ceil,
    'floor': math.floor,
    'factorial': math.factorial,
}

# Constantes matemáticas seguras
SAFE_CONSTANTS = {
    'pi': math.pi,
    'e': math.e,
    'tau': math.tau,
}


def _eval_node(node: ast.AST, variables: dict = None) -> Any:
    """
    Evalúa un nodo AST de forma segura.
    
    Args:
        node: Nodo AST a evaluar
        variables: Variables disponibles (por defecto incluye constantes matemáticas)
    
    Returns:
        Resultado de la evaluación
    
    Raises:
        ValueError: Si se encuentra una operación no permitida
    """
    if variables is None:
        variables = SAFE_CONSTANTS.copy()
    
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)
    elif isinstance(node, ast.Constant):  # Python 3.8+
        return node.value
    elif isinstance(node, ast.Num):  # Python < 3.8
        return node.n
    elif isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        elif node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        else:
            raise ValueError(f"Variable no permitida: {node.id}")
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in SAFE_OPERATORS:
            raise ValueError(f"Operador no permitido: {type(node.op)}")
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
        return SAFE_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in SAFE_OPERATORS:
            raise ValueError(f"Operador unario no permitido: {type(node.op)}")
        operand = _eval_node(node.operand, variables)
        return SAFE_OPERATORS[type(node.op)](operand)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Solo se permiten llamadas a funciones simples")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise ValueError(f"Función no permitida: {func_name}")
        args = [_eval_node(arg, variables) for arg in node.args]
        return SAFE_FUNCTIONS[func_name](*args)
    else:
        raise ValueError(f"Nodo AST no permitido: {type(node)}")


def eval_safe_expression(expr: str, variables: dict = None) -> float:
    """
    Evalúa una expresión matemática de forma segura.
    
    Args:
        expr: Expresión matemática como string
        variables: Variables adicionales disponibles
    
    Returns:
        Resultado numérico de la evaluación
    
    Raises:
        ValueError: Si la expresión no es válida o contiene operaciones no permitidas
    
    Examples:
        >>> eval_safe_expression("pi/2")
        1.5707963267948966
        >>> eval_safe_expression("sin(pi/4)")
        0.7071067811865475
        >>> eval_safe_expression("2*e + sqrt(16)")
        9.43656365691809
    """
    try:
        # Reemplazar 'x' por el valor si se proporciona
        if variables and 'x' in variables:
            expr = expr.replace('x', str(variables['x']))
        
        # Parsear la expresión
        tree = ast.parse(expr, mode='eval')
        
        # Evaluar de forma segura
        result = _eval_node(tree, variables)
        
        # Convertir a float si es posible
        return float(result)
        
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError) as e:
        raise ValueError(f"Error evaluando '{expr}': {str(e)}")


def make_safe_function(expr: str) -> Callable[[float], float]:
    """
    Crea una función segura a partir de una expresión matemática string.
    
    Args:
        expr: Expresión matemática que depende de 'x'
    
    Returns:
        Función que evalúa la expresión para valores dados de x
    
    Raises:
        ValueError: Si la expresión no es válida
    
    Examples:
        >>> f = make_safe_function("x**2 + sin(x)")
        >>> f(1.0)
        1.8414709848078965
        >>> f(0.0)
        0.0
    """
    def safe_function(x: float) -> float:
        """Función generada automáticamente desde expresión matemática"""
        try:
            return eval_safe_expression(expr, {'x': x})
        except Exception as e:
            raise ValueError(f"Error evaluando f({x}) = {expr}: {str(e)}")
    
    # Agregar metadatos útiles
    safe_function.__name__ = f"f(x) = {expr}"
    safe_function._expression = expr
    
    return safe_function


def validate_integration_parameters(a: float, b: float, n: int, tol: float = None) -> None:
    """
    Valida los parámetros de integración numérica.
    
    Args:
        a: Límite inferior
        b: Límite superior
        n: Número de subdivisiones
        tol: Tolerancia (opcional, para métodos adaptativos)
    
    Raises:
        ValueError: Si algún parámetro no es válido
    """
    if not isinstance(a, (int, float)) or not math.isfinite(a):
        raise ValueError(f"Límite inferior inválido: {a}")
    
    if not isinstance(b, (int, float)) or not math.isfinite(b):
        raise ValueError(f"Límite superior inválido: {b}")
    
    if a >= b:
        raise ValueError(f"El límite inferior ({a}) debe ser menor que el superior ({b})")
    
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"Número de subdivisiones debe ser entero positivo: {n}")
    
    if n > 10000:
        raise ValueError(f"Número de subdivisiones demasiado grande: {n} (máximo: 10000)")
    
    if tol is not None:
        if not isinstance(tol, (int, float)) or tol <= 0:
            raise ValueError(f"Tolerancia debe ser número positivo: {tol}")
        if tol > 1.0:
            raise ValueError(f"Tolerancia muy grande: {tol} (máximo recomendado: 1.0)")


def format_result(value: float, precision: int = 12) -> str:
    """
    Formatea un resultado numérico con la precisión especificada.
    
    Args:
        value: Valor a formatear
        precision: Número de decimales significativos
    
    Returns:
        String formateado del número
    
    Examples:
        >>> format_result(1.2345678901234567)
        '1.234567890123'
        >>> format_result(0.000123456789)
        '0.000123456789'
    """
    if abs(value) < 1e-15:
        return "0.0"
    
    # Usar notación científica para números muy grandes o muy pequeños
    if abs(value) >= 1e6 or abs(value) < 1e-4:
        return f"{value:.{precision-1}e}"
    else:
        return f"{value:.{precision}g}"


def calculate_step_size(a: float, b: float, n: int) -> float:
    """
    Calcula el tamaño de paso para integración numérica.
    
    Args:
        a: Límite inferior
        b: Límite superior  
        n: Número de subdivisiones
    
    Returns:
        Tamaño de paso h = (b-a)/n
    """
    return (b - a) / n


def generate_points(a: float, b: float, n: int) -> list:
    """
    Genera puntos equidistantes en el intervalo [a, b].
    
    Args:
        a: Límite inferior
        b: Límite superior
        n: Número de subdivisiones (genera n+1 puntos)
    
    Returns:
        Lista de puntos [x_0, x_1, ..., x_n]
    """
    h = calculate_step_size(a, b, n)
    return [a + i * h for i in range(n + 1)]


# Alias para compatibilidad con código existente
_eval_safe_expression = eval_safe_expression
_make_safe_func = make_safe_function
