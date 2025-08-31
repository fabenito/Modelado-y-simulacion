"""
Componentes principales de la interfaz gráfica del simulador.
Ventana principal y formularios de entrada.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Callable, Any
import math

from ..utils import eval_safe_expression, make_safe_function, check_for_singularities
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
        
        # Ventana más grande y redimensionable
        self.root.geometry("1200x800")
        self.root.minsize(1000, 600)
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
        """Configura la interfaz de usuario con mejor layout."""
        # Crear PanedWindow para dividir controles y visualización
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Frame izquierdo para controles (scrollable)
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame, weight=1)
        
        # Frame derecho para visualización
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame, weight=2)
        
        # Agregar scroll al frame izquierdo
        self._setup_left_scroll()
        
        # Crear secciones
        self._create_input_section()
        self._create_method_buttons()
        self._create_utility_buttons()
        self._create_result_section()
        
    def _setup_left_scroll(self):
        """Configura scroll para el panel izquierdo."""
        # Canvas para scroll en panel izquierdo
        self.left_canvas = tk.Canvas(self.left_frame, bg='lightgray')
        self.left_scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", 
                                          command=self.left_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.left_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))
        )
        
        self.left_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)
        
        # Pack
        self.left_canvas.pack(side="left", fill="both", expand=True)
        self.left_scrollbar.pack(side="right", fill="y")
        
        # Mousewheel binding para scroll suave
        def _on_mousewheel(event):
            self.left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _create_input_section(self):
        """Crea la sección de entrada de parámetros."""
        input_frame = ttk.LabelFrame(self.scrollable_frame, text="Parámetros de Integración", 
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
        
        # Ayuda y ejemplos
        help_text = ("Ejemplos normales: x**2, sin(x), exp(x), log(x)\n"
                    "Ejemplos L'Hôpital: sin(x)/x, (exp(x)-1)/x, (1-cos(x))/x**2\n"
                    "Límites: números o expresiones como 'pi/2', 'e', 'sqrt(2)'")
        ttk.Label(input_frame, text=help_text, foreground="gray", 
                 font=('Arial', 8)).grid(row=5, column=0, columnspan=2, 
                                       sticky="w", padx=5, pady=2)
        
        # Botones de ejemplo rápido
        examples_frame = ttk.Frame(input_frame)
        examples_frame.grid(row=6, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        ttk.Label(examples_frame, text="Ejemplos rápidos:", 
                 font=('Arial', 8, 'bold')).pack(side=tk.LEFT)
        
        examples = [
            ("x²", "x**2", "0", "1"),
            ("sin(x)/x", "sin(x)/x", "0", "pi"),
            ("(e^x-1)/x", "(exp(x)-1)/x", "0", "1"),
        ]
        
        for i, (label, expr, a_val, b_val) in enumerate(examples):
            ttk.Button(examples_frame, text=label, width=8,
                      command=lambda e=expr, a=a_val, b=b_val: self._set_example(e, a, b)
                      ).pack(side=tk.LEFT, padx=2)
    
    def _create_method_buttons(self):
        """Crea los botones para cada método de integración."""
        methods_frame = ttk.LabelFrame(self.scrollable_frame, text="Métodos de Integración", 
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
        
        # Resultados en el panel izquierdo (scrollable)
        self.result_display = ResultDisplay(self.scrollable_frame)
        
        # Visualización en el panel derecho (más espacio)
        self.visualization = IntegrationVisualization(self.right_frame)
    
    def _create_utility_buttons(self):
        """Crea botones utilitarios."""
        utils_frame = ttk.Frame(self.scrollable_frame)
        utils_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(utils_frame, text="Limpiar Tabla", 
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(utils_frame, text="Ver Fórmulas", 
                  command=self.show_formulas).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(utils_frame, text="Comparar Métodos", 
                  command=self.compare_all_methods).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(utils_frame, text="Ayuda L'Hôpital", 
                  command=self.show_lhopital_help).pack(side=tk.LEFT, padx=5)
    
    def run_method(self, method_key: str):
        """
        Ejecuta un método de integración específico con detección de singularidades.
        
        Args:
            method_key: Clave del método a ejecutar
        """
        try:
            # Obtener parámetros básicos
            expr = self.expr_var.get()
            a = eval_safe_expression(self.a_var.get())
            b = eval_safe_expression(self.b_var.get())
            
            # Verificar singularidades en los límites de integración
            has_singularities, critical_points, explanation = check_for_singularities(expr, a, b)
            
            # Crear función con manejo de L'Hôpital si es necesario
            lhopital_points = {}
            if has_singularities and critical_points:
                # Mostrar información sobre singularidades detectadas
                message = (
                    "🔍 Análisis de Singularidades\n"
                    "="+ "\n\n"
                    f"{explanation}\n\n"
                    "¿Desea aplicar la regla de L'Hôpital para resolver estas singularidades?\n\n"
                    "• SÍ: Usar los valores límite calculados\n"
                    "• NO: Intentar integración sin L'Hôpital (puede fallar)\n"
                    "• CANCELAR: Abortar cálculo"
                )
                
                # Usar messagebox personalizado con 3 opciones
                response = messagebox.askyesnocancel(
                    "Singularidades Detectadas - Regla de L'Hôpital", 
                    message
                )
                
                if response is None:  # Cancelar
                    return
                elif response:  # Sí, aplicar L'Hôpital
                    lhopital_points = {x: limit_val for x, limit_val in critical_points}
                    messagebox.showinfo(
                        "L'Hôpital Aplicado", 
                        f"Se aplicará L'Hôpital en {len(critical_points)} punto(s):\n" +
                        "\n".join([f"x={x}: límite={lim}" for x, lim in critical_points])
                    )
                # Si respuesta es False (No), continuar sin L'Hôpital
            
            # Crear función (con o sin L'Hôpital según la decisión del usuario)
            func = make_safe_function(expr, lhopital_points)
            
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
            
            # Mostrar información adicional si se usó L'Hôpital
            if lhopital_points:
                lhopital_info = (
                    f"✅ Integración completada con L'Hôpital aplicado en:\n" +
                    "\n".join([f"  • x={x}: f(x) = {lim}" for x, lim in lhopital_points.items()])
                )
                messagebox.showinfo("L'Hôpital Aplicado", lhopital_info)
            
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
        """Compara todos los métodos disponibles con manejo de singularidades."""
        try:
            from ..methods import compare_methods
            
            expr = self.expr_var.get()
            a = eval_safe_expression(self.a_var.get())
            b = eval_safe_expression(self.b_var.get())
            n = int(self.n_var.get())
            
            # Verificar singularidades
            has_singularities, critical_points, explanation = check_for_singularities(expr, a, b)
            
            # Manejar L'Hôpital si es necesario
            lhopital_points = {}
            if has_singularities and critical_points:
                message = (
                    "🔍 Singularidades detectadas para comparación\n"
                    "=" * 45 + "\n\n"
                    f"{explanation}\n\n"
                    "Para comparar métodos de forma consistente:\n"
                    "¿Aplicar L'Hôpital en todos los métodos?"
                )
                
                response = messagebox.askyesno("L'Hôpital para Comparación", message)
                if response:
                    lhopital_points = {x: limit_val for x, limit_val in critical_points}
            
            # Crear función
            func = make_safe_function(expr, lhopital_points)
            
            # Intentar calcular valor exacto para funciones simples
            exact_value = self._try_exact_integration(expr, a, b)
            
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
    
    def show_lhopital_help(self):
        """Muestra información sobre el manejo de singularidades con L'Hôpital."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Ayuda: Regla de L'Hôpital")
        help_window.geometry("700x500")
        help_window.configure(bg='white')
        
        # Crear frame con scroll
        frame = tk.Frame(help_window, bg='white')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Texto de ayuda
        help_text = """
🧮 REGLA DE L'HÔPITAL EN INTEGRACIÓN NUMÉRICA
═══════════════════════════════════════════════

¿Qué es una singularidad removible?
───────────────────────────────────────────────
Una singularidad removible ocurre cuando una función no está definida 
en un punto, pero tiene un límite finito cuando nos aproximamos a ese punto.

Ejemplo clásico: f(x) = sin(x)/x
• En x=0: sin(0)/0 = 0/0 (indefinido)
• Pero lim(x→0) sin(x)/x = 1 (usando L'Hôpital)

¿Cuándo aparece en integración numérica?
───────────────────────────────────────────────
• Límite inferior = 0 y función con forma sin(x)/x
• Funciones con denominadores que se anulan en los límites
• Formas indeterminadas 0/0 o ∞/∞

Funciones compatibles detectadas automáticamente:
───────────────────────────────────────────────
✅ sin(x)/x        → límite en x=0: 1
✅ sinh(x)/x       → límite en x=0: 1  
✅ tan(x)/x        → límite en x=0: 1
✅ (1-cos(x))/x²   → límite en x=0: 1/2
✅ (eˣ-1)/x        → límite en x=0: 1
✅ log(1+x)/x      → límite en x=0: 1

¿Cómo funciona en el simulador?
───────────────────────────────────────────────
1. 🔍 El sistema detecta automáticamente singularidades
2. 📊 Calcula el límite usando L'Hôpital cuando es posible
3. ❓ Pregunta al usuario si desea aplicar la corrección
4. ✅ Reemplaza el valor problemático con el límite calculado
5. 📈 Continúa la integración normalmente

Ejemplo de uso:
───────────────────────────────────────────────
Función: sin(x)/x
Límites: [0, π]
Subdivisions: 10

Sin L'Hôpital → Error: División por cero en x=0
Con L'Hôpital → f(0) = 1, integración exitosa

Nota: Si rechaza usar L'Hôpital, el cálculo puede fallar o dar 
resultados incorrectos debido a la división por cero.
        """
        
        text_widget = tk.Text(frame, wrap=tk.WORD, font=('Courier New', 10), 
                            bg='white', fg='black', relief='flat')
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Solo lectura
        
        # Scrollbar para el texto
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Botón cerrar
        ttk.Button(help_window, text="Cerrar", 
                  command=help_window.destroy).pack(pady=10)
    
    def _set_example(self, expr: str, a_val: str, b_val: str):
        """Establece un ejemplo en los campos de entrada."""
        self.expr_var.set(expr)
        self.a_var.set(a_val)
        self.b_var.set(b_val)
    
    def run(self):
        """Inicia el bucle principal de la aplicación."""
        self.root.mainloop()
    
    def get_root(self) -> tk.Tk:
        """Retorna la ventana raíz para componentes hijos."""
        return self.root
