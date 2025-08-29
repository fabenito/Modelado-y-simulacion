import ast
import math
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional, List, Tuple
    
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    import numpy as np
except Exception:
    FigureCanvasTkAgg = None
    plt = None
    np = None


def _make_safe_func(expr: str) -> Callable[[float], float]:
    """Crea una función segura a partir de una expresión matemática string"""
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    allowed_names.update({"abs": abs, "pow": pow})
    expr_ast = ast.parse(expr, mode='eval')
    for node in ast.walk(expr_ast):
        if isinstance(node, ast.Name):
            if node.id != 'x' and node.id not in allowed_names:
                raise ValueError(f"Nombre no permitido en expresión: {node.id}")
        elif isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp, ast.Expression,
                               ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
                               ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.Constant,
                               ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.Gt,
                               ast.LtE, ast.GtE, ast.And, ast.Or, ast.BoolOp)):
            continue
        else:
            raise ValueError(f"Nodo AST no permitido: {type(node).__name__}")
    code = compile(expr_ast, '<string>', 'eval')
    def f(x: float) -> float:
        return eval(code, {'__builtins__': {}}, {**allowed_names, 'x': x})
    return f


def _eval_safe_expression(expr: str) -> float:
    """Evalúa una expresión matemática segura y devuelve un valor numérico"""
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
    allowed_names.update({"abs": abs, "pow": pow})
    
    # Primero intentar convertir directamente a float
    try:
        return float(expr)
    except ValueError:
        pass
    
    # Si no es un número simple, evaluar como expresión matemática
    try:
        expr_ast = ast.parse(expr, mode='eval')
        for node in ast.walk(expr_ast):
            if isinstance(node, ast.Name):
                if node.id not in allowed_names:
                    raise ValueError(f"Nombre no permitido en expresión: {node.id}")
            elif isinstance(node, (ast.Call, ast.BinOp, ast.UnaryOp, ast.Expression,
                                   ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
                                   ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.Constant,
                                   ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.Gt,
                                   ast.LtE, ast.GtE, ast.And, ast.Or, ast.BoolOp)):
                continue
            else:
                raise ValueError(f"Nodo AST no permitido: {type(node).__name__}")
        code = compile(expr_ast, '<string>', 'eval')
        return float(eval(code, {'__builtins__': {}}, allowed_names))
    except Exception as e:
        raise ValueError(f"Error al evaluar la expresión '{expr}': {str(e)}")


def regla_trapezoidal(f: Callable[[float], float], a: float, b: float, n: int = 1):
    """Regla trapezoidal (Newton-Cotes de grado 1)"""
    h = (b - a) / n
    integral = 0.5 * (f(a) + f(b))
    
    history = [(0, a, f(a), 0.5 * f(a))]
    
    for i in range(1, n):
        x_i = a + i * h
        f_xi = f(x_i)
        integral += f_xi
        history.append((i, x_i, f_xi, f_xi))
    
    history.append((n, b, f(b), 0.5 * f(b)))
    integral *= h
    
    return integral, history, h


def regla_simpson_1_3(f: Callable[[float], float], a: float, b: float, n: int = 2):
    """Regla de Simpson 1/3 (Newton-Cotes de grado 2)"""
    if n % 2 != 0:
        n += 1  # Asegurar que n sea par
    
    h = (b - a) / n
    integral = f(a) + f(b)
    history = [(0, a, f(a), 1), (n, b, f(b), 1)]
    
    # Puntos impares (coeficiente 4)
    for i in range(1, n, 2):
        x_i = a + i * h
        f_xi = f(x_i)
        integral += 4 * f_xi
        history.append((i, x_i, f_xi, 4))
    
    # Puntos pares (coeficiente 2)
    for i in range(2, n, 2):
        x_i = a + i * h
        f_xi = f(x_i)
        integral += 2 * f_xi
        history.append((i, x_i, f_xi, 2))
    
    integral *= h / 3
    history.sort(key=lambda x: x[0])  # Ordenar por índice
    
    return integral, history, h


def regla_simpson_3_8(f: Callable[[float], float], a: float, b: float, n: int = 3):
    """Regla de Simpson 3/8 (Newton-Cotes de grado 3)"""
    if n % 3 != 0:
        n = 3 * ((n // 3) + 1)  # Asegurar que n sea múltiplo de 3
    
    h = (b - a) / n
    integral = f(a) + f(b)
    history = [(0, a, f(a), 1), (n, b, f(b), 1)]
    
    for i in range(1, n):
        x_i = a + i * h
        f_xi = f(x_i)
        if i % 3 == 0:
            coeff = 2
        else:
            coeff = 3
        integral += coeff * f_xi
        history.append((i, x_i, f_xi, coeff))
    
    integral *= 3 * h / 8
    history.sort(key=lambda x: x[0])
    
    return integral, history, h


def regla_boole(f: Callable[[float], float], a: float, b: float, n: int = 4):
    """Regla de Boole (Newton-Cotes de grado 4)"""
    if n % 4 != 0:
        n = 4 * ((n // 4) + 1)  # Asegurar que n sea múltiplo de 4
    
    h = (b - a) / n
    coefficients = [7, 32, 12, 32, 7]  # Patrón que se repite cada 4 intervalos
    
    integral = 0
    history = []
    
    for i in range(n + 1):
        x_i = a + i * h
        f_xi = f(x_i)
        coeff = coefficients[i % 4] if i % 4 != 0 else (7 if i == 0 or i == n else 14)
        if i == 0 or i == n:
            coeff = 7
        elif i % 4 == 1 or i % 4 == 3:
            coeff = 32
        elif i % 4 == 2:
            coeff = 12
        else:
            coeff = 14  # Puntos que son múltiplos de 4 pero no extremos
            
        integral += coeff * f_xi
        history.append((i, x_i, f_xi, coeff))
    
    integral *= 2 * h / 45
    
    return integral, history, h


def integracion_adaptativa_simpson(f: Callable[[float], float], a: float, b: float, 
                                 tol: float = 1e-8, max_depth: int = 10):
    """Integración adaptativa usando Simpson con control de error"""
    def simpson_basico(f, a, b):
        h = (b - a) / 2
        return h / 3 * (f(a) + 4 * f(a + h) + f(b))
    
    def adaptativa_recursiva(f, a, b, tol, S_ab, depth):
        if depth > max_depth:
            return S_ab, [(depth, a, b, S_ab, 0)]
        
        c = (a + b) / 2
        S_ac = simpson_basico(f, a, c)
        S_cb = simpson_basico(f, c, b)
        S_acb = S_ac + S_cb
        
        error = abs(S_acb - S_ab) / 15  # Error estimado
        
        if error < tol:
            return S_acb, [(depth, a, b, S_acb, error)]
        else:
            left_result, left_hist = adaptativa_recursiva(f, a, c, tol/2, S_ac, depth + 1)
            right_result, right_hist = adaptativa_recursiva(f, c, b, tol/2, S_cb, depth + 1)
            return left_result + right_result, left_hist + right_hist
    
    S_initial = simpson_basico(f, a, b)
    result, history = adaptativa_recursiva(f, a, b, tol, S_initial, 0)
    
    return result, history


class IntegracionGUI:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Métodos Numéricos - Integración (Newton-Cotes)")
        master.geometry("1000x800")
        self._build_widgets()

    def _build_widgets(self):
        frm = ttk.Frame(self.master, padding=10)
        frm.grid(row=0, column=0, sticky='nsew')
        
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        frm.grid_rowconfigure(5, weight=1)
        frm.grid_columnconfigure(0, weight=1)

        # Función a integrar
        ttk.Label(frm, text="f(x):").grid(row=0, column=0, sticky='w')
        self.expr_var = tk.StringVar(value="sin(x)")
        ttk.Entry(frm, textvariable=self.expr_var, width=50).grid(row=0, column=1, columnspan=3, sticky='we')

        # Límites de integración
        ttk.Label(frm, text="Límite inferior (a):").grid(row=1, column=0, sticky='w')
        self.a_var = tk.StringVar(value="0")
        ttk.Entry(frm, textvariable=self.a_var, width=15).grid(row=1, column=1, sticky='w')

        ttk.Label(frm, text="Límite superior (b):").grid(row=1, column=2, sticky='w')
        self.b_var = tk.StringVar(value="pi/2")
        ttk.Entry(frm, textvariable=self.b_var, width=15).grid(row=1, column=3, sticky='w')

        # Etiqueta de ayuda para los límites
        help_label = ttk.Label(frm, text="Tip: Puedes usar expresiones como pi, pi/2, e, sqrt(2), etc.", 
                              font=('TkDefaultFont', 8), foreground='gray')
        help_label.grid(row=1, column=4, columnspan=2, sticky='w', padx=(10,0))

        # Número de subdivisiones
        ttk.Label(frm, text="Subdivisiones (n):").grid(row=2, column=0, sticky='w')
        self.n_var = tk.StringVar(value="10")
        ttk.Entry(frm, textvariable=self.n_var, width=10).grid(row=2, column=1, sticky='w')

        # Tolerancia para método adaptativo
        ttk.Label(frm, text="Tolerancia (adaptativo):").grid(row=2, column=2, sticky='w')
        self.tol_var = tk.StringVar(value="1e-6")
        ttk.Entry(frm, textvariable=self.tol_var, width=10).grid(row=2, column=3, sticky='w')

        # Botones de métodos
        button_frame = ttk.Frame(frm)
        button_frame.grid(row=3, column=0, columnspan=4, pady=10, sticky='we')
        
        ttk.Button(button_frame, text="Trapezoidal", command=self.run_trapezoidal).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Simpson 1/3", command=self.run_simpson_1_3).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Simpson 3/8", command=self.run_simpson_3_8).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Boole", command=self.run_boole).grid(row=0, column=3, padx=5)
        ttk.Button(button_frame, text="Adaptativo", command=self.run_adaptativo).grid(row=0, column=4, padx=5)
        ttk.Button(button_frame, text="Limpiar", command=self.clear_all).grid(row=0, column=5, padx=5)

        # Tabla de resultados
        cols = ("i", "x_i", "f(x_i)", "coef", "contribución")
        self.tree = ttk.Treeview(frm, columns=cols, show='headings', height=8)
        col_widths = {"i": 50, "x_i": 140, "f(x_i)": 140, "coef": 60, "contribución": 140}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=col_widths[c], anchor='center')
        
        scrollbar = ttk.Scrollbar(frm, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.grid(row=5, column=0, columnspan=4, sticky='nsew', pady=10)
        scrollbar.grid(row=5, column=4, sticky='ns', pady=10)

        # Resultado
        self.result_var = tk.StringVar(value="Resultado: -")
        result_label = ttk.Label(frm, textvariable=self.result_var, font=('TkDefaultFont', 10, 'bold'))
        result_label.grid(row=6, column=0, columnspan=4, sticky='w', pady=5)

        # Gráfico directamente en el frame principal (como en tu simulador de raíces)
        if FigureCanvasTkAgg and plt:
            self.fig, self.ax = plt.subplots(figsize=(8, 4))
            self.canvas = FigureCanvasTkAgg(self.fig, master=frm)
            self.canvas.get_tk_widget().grid(row=7, column=0, columnspan=5, pady=10)
        else:
            self.canvas = None
            self.ax = None
            ttk.Label(frm, text="matplotlib no disponible").grid(row=7, column=0, columnspan=5)

    def _populate_table(self, history, h, method='trapezoidal'):
        """Llena la tabla con los puntos de evaluación"""
        for row in self.tree.get_children():
            self.tree.delete(row)
        
        for i, x_i, f_xi, coef in history:
            contrib = h * coef * f_xi / (3 if 'simpson' in method else (8/3 if method == 'simpson_3_8' else (2/45 if method == 'boole' else 1)))
            values = (i, f"{x_i:.10g}", f"{f_xi:.10g}", coef, f"{contrib:.10g}")
            self.tree.insert('', 'end', values=values)

    def _populate_adaptive_table(self, history):
        """Llena la tabla para el método adaptativo"""
        for row in self.tree.get_children():
            self.tree.delete(row)
            
        # Cambiar headers para método adaptativo
        self.tree.heading("i", text="Nivel")
        self.tree.heading("x_i", text="a")
        self.tree.heading("f(x_i)", text="b")
        self.tree.heading("coef", text="Integral")
        self.tree.heading("contribución", text="Error Est.")
        
        for depth, a, b, integral, error in history:
            values = (depth, f"{a:.10g}", f"{b:.10g}", f"{integral:.10g}", f"{error:.6e}")
            self.tree.insert('', 'end', values=values)

    def _plot(self, func, a, b, history=None, method='trapezoidal'):
        """Grafica la función y la aproximación numérica con visualización del área"""
        if not self.ax or not self.canvas:
            return
            
        self.ax.clear()
        
        # Graficar la función original
        x_vals = [a + i * (b - a) / 1000 for i in range(1001)]
        y_vals = [func(x) for x in x_vals]
        self.ax.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='f(x)', zorder=3)
        
        # Sombrear el área real bajo la curva para referencia
        self.ax.fill_between(x_vals, 0, y_vals, alpha=0.1, color='blue', label='Área real')
        
        if history:
            x_points = [point[1] for point in history]
            y_points = [point[2] for point in history]
            
            # Dibujar aproximación específica según el método
            if method == 'trapezoidal':
                self._plot_trapezoidal(x_points, y_points, func)
            elif method == 'simpson_1_3':
                self._plot_simpson(x_points, y_points, func, method='1/3')
            elif method == 'simpson_3_8':
                self._plot_simpson(x_points, y_points, func, method='3/8')
            elif method == 'boole':
                self._plot_boole(x_points, y_points, func)
            else:
                # Método genérico para otros casos
                self.ax.plot(x_points, y_points, 'ro', markersize=6, label='Puntos de evaluación', zorder=4)
                self.ax.fill_between(x_points, 0, y_points, alpha=0.3, color='red', 
                                   step='pre', label=f'Aproximación {method}')
        
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3, zorder=1)
        self.ax.grid(True, linestyle='--', alpha=0.4, zorder=1)
        self.ax.legend(loc='best')
        self.ax.set_xlabel('x', fontsize=11)
        self.ax.set_ylabel('f(x)', fontsize=11)
        self.ax.set_title(f'Integración Numérica - {method.replace("_", " ").title()}', fontsize=12, pad=15)
        
        # Ajustar límites del gráfico
        y_min = min(min(y_vals), 0)
        y_max = max(max(y_vals), 0)
        margin = (y_max - y_min) * 0.1
        self.ax.set_ylim(y_min - margin, y_max + margin)
        
        self.canvas.draw()
        
    def _plot_trapezoidal(self, x_points, y_points, func):
        """Visualización específica para la regla trapezoidal"""
        # Dibujar puntos de evaluación
        self.ax.plot(x_points, y_points, 'ro', markersize=6, label='Puntos de evaluación', zorder=4)
        
        # Dibujar trapecios individuales
        for i in range(len(x_points) - 1):
            x_trap = [x_points[i], x_points[i], x_points[i+1], x_points[i+1]]
            y_trap = [0, y_points[i], y_points[i+1], 0]
            self.ax.fill(x_trap, y_trap, alpha=0.4, color='red', edgecolor='darkred', linewidth=1)
            
        # Líneas de conexión (aproximación trapezoidal)
        self.ax.plot(x_points, y_points, 'r--', alpha=0.8, linewidth=1.5, 
                    label='Aproximación trapezoidal', zorder=3)
        
    def _plot_simpson(self, x_points, y_points, func, method='1/3'):
        """Visualización específica para las reglas de Simpson"""
        # Dibujar puntos de evaluación
        self.ax.plot(x_points, y_points, 'go', markersize=6, label='Puntos de evaluación', zorder=4)
        
        # Para Simpson, crear una aproximación más suave usando interpolación parabólica
        if method == '1/3' and len(x_points) >= 3:
            # Procesar cada par de intervalos (parabólica)
            for i in range(0, len(x_points)-2, 2):
                if i+2 < len(x_points):
                    x_seg = x_points[i:i+3]
                    y_seg = y_points[i:i+3]
                    
                    # Crear interpolación parabólica suave
                    x_smooth = [x_seg[0] + j * (x_seg[2] - x_seg[0]) / 50 for j in range(51)]
                    # Interpolación cuadrática simple
                    y_smooth = []
                    for x in x_smooth:
                        # Interpolación de Lagrange de 2do grado
                        y = (y_seg[0] * (x - x_seg[1]) * (x - x_seg[2]) / 
                             ((x_seg[0] - x_seg[1]) * (x_seg[0] - x_seg[2])) +
                             y_seg[1] * (x - x_seg[0]) * (x - x_seg[2]) / 
                             ((x_seg[1] - x_seg[0]) * (x_seg[1] - x_seg[2])) +
                             y_seg[2] * (x - x_seg[0]) * (x - x_seg[1]) / 
                             ((x_seg[2] - x_seg[0]) * (x_seg[2] - x_seg[1])))
                        y_smooth.append(max(0, y))  # Evitar valores negativos en el área
                    
                    # Sombrear área bajo la parábola
                    self.ax.fill_between(x_smooth, 0, y_smooth, alpha=0.4, color='green')
                    
                    if i == 0:  # Solo agregar label una vez
                        self.ax.plot(x_smooth, y_smooth, 'g--', alpha=0.8, linewidth=1.5,
                                   label=f'Aproximación Simpson {method}', zorder=3)
                    else:
                        self.ax.plot(x_smooth, y_smooth, 'g--', alpha=0.8, linewidth=1.5, zorder=3)
        else:
            # Fallback para casos donde no se puede hacer interpolación parabólica
            self.ax.fill_between(x_points, 0, y_points, alpha=0.4, color='green',
                               step='pre', label=f'Aproximación Simpson {method}')
            
    def _plot_boole(self, x_points, y_points, func):
        """Visualización específica para la regla de Boole"""
        # Dibujar puntos de evaluación
        self.ax.plot(x_points, y_points, 'mo', markersize=6, label='Puntos de evaluación', zorder=4)
        
        # Para Boole, procesar cada grupo de 5 puntos (polinomio de 4to grado)
        if len(x_points) >= 5:
            for i in range(0, len(x_points)-4, 4):
                if i+4 < len(x_points):
                    x_seg = x_points[i:i+5]
                    y_seg = y_points[i:i+5]
                    
                    # Crear una aproximación suave (simplificada)
                    x_smooth = [x_seg[0] + j * (x_seg[4] - x_seg[0]) / 20 for j in range(21)]
                    # Interpolación lineal por segmentos como aproximación
                    y_smooth = [func(x) for x in x_smooth]
                    
                    # Sombrear área
                    self.ax.fill_between(x_smooth, 0, y_smooth, alpha=0.4, color='magenta')
                    
                    if i == 0:  # Solo agregar label una vez
                        self.ax.plot(x_smooth, y_smooth, 'm--', alpha=0.8, linewidth=1.5,
                                   label='Aproximación Boole', zorder=3)
                    else:
                        self.ax.plot(x_smooth, y_smooth, 'm--', alpha=0.8, linewidth=1.5, zorder=3)
        else:
            # Fallback
            self.ax.fill_between(x_points, 0, y_points, alpha=0.4, color='magenta',
                               step='pre', label='Aproximación Boole')

    def clear_all(self):
        """Limpia todos los resultados"""
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.result_var.set("Resultado: -")
        
        # Restaurar headers originales
        cols = ("i", "x_i", "f(x_i)", "coef", "contribución")
        for c in cols:
            self.tree.heading(c, text=c)
        
        if self.ax and self.canvas:
            self.ax.clear()
            self.canvas.draw()

    def _run_method(self, method_func, method_name):
        """Método genérico para ejecutar cualquier método de integración"""
        try:
            f = _make_safe_func(self.expr_var.get())
            a = _eval_safe_expression(self.a_var.get())
            b = _eval_safe_expression(self.b_var.get())
            n = int(self.n_var.get())
        except Exception as e:
            messagebox.showerror("Error", f"Error en los parámetros: {str(e)}")
            return
            
        try:
            if method_name == 'adaptativo':
                tol = float(self.tol_var.get())
                result, history = integracion_adaptativa_simpson(f, a, b, tol)
                self._populate_adaptive_table(history)
            else:
                result, history, h = method_func(f, a, b, n)
                self._populate_table(history, h, method_name)
                
            self.result_var.set(f"Integral ≈ {result:.12g}")
            
            if self.canvas and plt:
                self._plot(f, a, b, history if method_name != 'adaptativo' else None, method_name)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error en el cálculo: {str(e)}")

    def run_trapezoidal(self):
        self._run_method(regla_trapezoidal, 'trapezoidal')

    def run_simpson_1_3(self):
        self._run_method(regla_simpson_1_3, 'simpson_1_3')

    def run_simpson_3_8(self):
        self._run_method(regla_simpson_3_8, 'simpson_3_8')

    def run_boole(self):
        self._run_method(regla_boole, 'boole')

    def run_adaptativo(self):
        self._run_method(None, 'adaptativo')


def main():
    root = tk.Tk()
    IntegracionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
