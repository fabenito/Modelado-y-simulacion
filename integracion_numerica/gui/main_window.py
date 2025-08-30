"""
Componentes principales de la interfaz gráfica del simulador.
Ventana principal y formularios de entrada.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Any
import math

from ..utils import eval_safe_expression, make_safe_function
from ..methods import get_all_methods


class MainWindow:
    """
    Ventana principal del simulador de integración numérica.
    Contiene formularios de entrada y controles principales.
    """
    
    def __init__(self):
        """Inicializa la ventana principal."""
        self.root = tk.Tk()
        self.root.title("Simulador de Integración Numérica - Métodos Newton-Cotes")
        self.root.geometry("900x700")
        self.root.configure(bg='lightgray')
        
        # Variables de entrada
        self.expr_var = tk.StringVar(value="x**2")
        self.a_var = tk.StringVar(value="0")
        self.b_var = tk.StringVar(value="1")
        self.n_var = tk.StringVar(value="10")
        self.tol_var = tk.StringVar(value="1e-6")
        
        # Referencias a componentes
        self.result_display = None
        self.visualization = None
        self.formula_display = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura la interfaz de usuario."""
        self._create_input_section()
        self._create_method_buttons()
        self._create_result_section()
        self._create_utility_buttons()
    
    def _create_input_section(self):
        """Crea la sección de entrada de parámetros."""
        input_frame = ttk.LabelFrame(self.root, text="Parámetros de Integración", 
                                   padding="10")
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Función
        ttk.Label(input_frame, text="Función f(x):").grid(row=0, column=0, 
                                                          sticky="w", padx=5, pady=2)
        func_entry = ttk.Entry(input_frame, textvariable=self.expr_var, width=30)
        func_entry.grid(row=0, column=1, padx=5, pady=2)
        
        # Límites
        ttk.Label(input_frame, text="Límite inferior (a):").grid(row=1, column=0, 
                                                                sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.a_var, width=15).grid(row=1, column=1, 
                                                                      sticky="w", padx=5, pady=2)
        
        ttk.Label(input_frame, text="Límite superior (b):").grid(row=2, column=0, 
                                                                sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.b_var, width=15).grid(row=2, column=1, 
                                                                      sticky="w", padx=5, pady=2)
        
        # Subdivisiones
        ttk.Label(input_frame, text="Subdivisiones (n):").grid(row=3, column=0, 
                                                              sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.n_var, width=15).grid(row=3, column=1, 
                                                                      sticky="w", padx=5, pady=2)
        
        # Tolerancia (para método adaptativo)
        ttk.Label(input_frame, text="Tolerancia (adaptativo):").grid(row=4, column=0, 
                                                                    sticky="w", padx=5, pady=2)
        ttk.Entry(input_frame, textvariable=self.tol_var, width=15).grid(row=4, column=1, 
                                                                        sticky="w", padx=5, pady=2)
        
        # Ayuda
        help_text = ("Ejemplos: x**2, sin(x), exp(x), log(x)\n"
                    "Límites: números o expresiones como 'pi/2', 'e', 'sqrt(2)'")
        ttk.Label(input_frame, text=help_text, foreground="gray", 
                 font=('Arial', 8)).grid(row=5, column=0, columnspan=2, 
                                       sticky="w", padx=5, pady=2)
    
    def _create_method_buttons(self):
        """Crea los botones para cada método de integración."""
        methods_frame = ttk.LabelFrame(self.root, text="Métodos de Integración", 
                                     padding="10")
        methods_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Obtener métodos disponibles
        methods = get_all_methods()
        
        # Información de botones: (key, text, color)
        button_info = [
            ('rectangulo', 'Rectángulo\n(Grado 0)', '#E0F7FA'),
            ('trapezoidal', 'Trapezoidal\n(Grado 1)', '#FFEBEE'),
            ('simpson_13', 'Simpson 1/3\n(Grado 2)', '#E8F5E8'),
            ('simpson_38', 'Simpson 3/8\n(Grado 3)', '#FFF3E0'),
            ('boole', 'Boole\n(Grado 4)', '#F3E5F5'),
            ('adaptativo', 'Adaptativo\n(Simpson)', '#E3F2FD')
        ]
        
        # Crear botones en 2 filas de 3
        for i, (method_key, button_text, bg_color) in enumerate(button_info):
            row = i // 3
            col = i % 3
            
            if method_key in methods:
                btn = tk.Button(
                    methods_frame,
                    text=button_text,
                    command=lambda k=method_key: self.run_method(k),
                    width=12,
                    height=2,
                    bg=bg_color,
                    relief="raised",
                    font=('Arial', 9, 'bold')
                )
                btn.grid(row=row, column=col, padx=8, pady=5)
    
    def _create_result_section(self):
        """Crea la sección de resultados y visualización."""
        # Import aquí para evitar dependencias circulares
        from .results import ResultDisplay
        from .visualization import IntegrationVisualization
        
        self.result_display = ResultDisplay(self.root)
        self.visualization = IntegrationVisualization(self.root)
    
    def _create_utility_buttons(self):
        """Crea botones utilitarios."""
        utils_frame = ttk.Frame(self.root)
        utils_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(utils_frame, text="Limpiar Tabla", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(utils_frame, text="Ver Fórmulas", 
                  command=self.show_formulas).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(utils_frame, text="Comparar Métodos", 
                  command=self.compare_all_methods).pack(side=tk.LEFT, padx=5)
    
    def run_method(self, method_key: str):
        """
        Ejecuta un método de integración específico.
        
        Args:
            method_key: Clave del método a ejecutar
        """
        try:
            # Validar y preparar parámetros
            func = make_safe_function(self.expr_var.get())
            a = eval_safe_expression(self.a_var.get())
            b = eval_safe_expression(self.b_var.get())
            
            # Obtener método
            methods = get_all_methods()
            if method_key not in methods:
                raise ValueError(f"Método '{method_key}' no disponible")
            
            method = methods[method_key]
            
            # Ejecutar método
            if method_key == 'adaptativo':
                tol = float(self.tol_var.get())
                method.tolerance = tol
                result = method.integrate(func, a, b)
            else:
                n = int(self.n_var.get())
                
                # Ajustar n según requerimientos del método
                if method_key == 'simpson_13' and n % 2 != 0:
                    n = n + 1
                    messagebox.showinfo("Ajuste", f"Simpson 1/3 requiere n par. Ajustado a n={n}")
                elif method_key == 'simpson_38' and n % 3 != 0:
                    n = ((n // 3) + 1) * 3
                    messagebox.showinfo("Ajuste", f"Simpson 3/8 requiere n múltiplo de 3. Ajustado a n={n}")
                elif method_key == 'boole' and n % 4 != 0:
                    n = ((n // 4) + 1) * 4
                    messagebox.showinfo("Ajuste", f"Boole requiere n múltiplo de 4. Ajustado a n={n}")
                
                result = method.integrate(func, a, b, n)
            
            # Mostrar resultados
            self.result_display.show_result(result, method_key)
            self.visualization.plot_method(func, a, b, result, method_key)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error ejecutando método: {str(e)}")
    
    def clear_results(self):
        """Limpia los resultados mostrados."""
        if self.result_display:
            self.result_display.clear()
        if self.visualization:
            self.visualization.clear()
    
    def show_formulas(self):
        """Muestra la ventana con fórmulas de integración."""
        from .formulas import FormulaDisplay
        FormulaDisplay(self.root)
    
    def compare_all_methods(self):
        """Compara todos los métodos disponibles."""
        try:
            from ..methods import compare_methods
            
            func = make_safe_function(self.expr_var.get())
            a = eval_safe_expression(self.a_var.get())
            b = eval_safe_expression(self.b_var.get())
            n = int(self.n_var.get())
            
            # Intentar calcular valor exacto para funciones simples
            exact_value = self._try_exact_integration(self.expr_var.get(), a, b)
            
            results = compare_methods(func, a, b, n, exact_value)
            self._show_comparison_results(results)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error en comparación: {str(e)}")
    
    def _try_exact_integration(self, expr: str, a: float, b: float) -> Optional[float]:
        """
        Intenta calcular el valor exacto para funciones simples.
        
        Returns:
            Valor exacto si se puede calcular, None en caso contrario
        """
        try:
            expr_clean = expr.strip().replace('**', '^')
            
            # Casos simples conocidos
            if expr_clean == 'x^2' or expr_clean == 'x**2':
                return (b**3 - a**3) / 3
            elif expr_clean == 'x':
                return (b**2 - a**2) / 2
            elif expr_clean == '1':
                return b - a
            elif expr_clean in ['sin(x)', 'math.sin(x)']:
                return -math.cos(b) + math.cos(a)
            elif expr_clean in ['cos(x)', 'math.cos(x)']:
                return math.sin(b) - math.sin(a)
            elif expr_clean in ['exp(x)', 'math.exp(x)']:
                return math.exp(b) - math.exp(a)
            
        except:
            pass
        
        return None
    
    def _show_comparison_results(self, results: dict):
        """Muestra los resultados de comparación en una ventana nueva."""
        comp_window = tk.Toplevel(self.root)
        comp_window.title("Comparación de Métodos")
        comp_window.geometry("600x400")
        
        # Crear tabla de resultados
        columns = ("Método", "Resultado", "Evaluaciones", "Error Abs.", "Error Rel.")
        tree = ttk.Treeview(comp_window, columns=columns, show="headings")
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        
        # Llenar tabla
        for method_name, data in results.items():
            if 'error' not in data:
                values = [
                    data.get('method', method_name),
                    f"{data['value']:.8f}",
                    str(data['evaluations']),
                    f"{data.get('absolute_error', 'N/A')}",
                    f"{data.get('relative_error', 'N/A')}"
                ]
                tree.insert("", tk.END, values=values)
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Button(comp_window, text="Cerrar", 
                  command=comp_window.destroy).pack(pady=10)
    
    def run(self):
        """Inicia el bucle principal de la aplicación."""
        self.root.mainloop()
    
    def get_root(self) -> tk.Tk:
        """Retorna la ventana raíz para componentes hijos."""
        return self.root
