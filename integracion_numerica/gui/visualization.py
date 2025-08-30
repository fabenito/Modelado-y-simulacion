"""
Componente de visualización para métodos de integración numérica.
Incluye gráficas de funciones y representaciones geométricas de los métodos.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional
import math

from ..methods.base import IntegrationResult
from ..utils import generate_points

# Importar matplotlib de forma segura
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class IntegrationVisualization:
    """
    Componente para visualizar métodos de integración numérica.
    """
    
    def __init__(self, parent: tk.Widget):
        """
        Inicializa el componente de visualización.
        
        Args:
            parent: Widget padre
        """
        self.parent = parent
        self.ax = None
        self.canvas = None
        self.fig = None
        
        if MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib_ui()
        else:
            self._setup_fallback_ui()
    
    def _setup_matplotlib_ui(self):
        """Configura la UI con matplotlib."""
        # Marco para la visualización
        self.viz_frame = ttk.LabelFrame(self.parent, text="Visualización", 
                                      padding="10")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Crear figura matplotlib
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Canvas para integrar matplotlib con tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, self.viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Configurar estilo inicial
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('f(x)', fontsize=12)
        self.ax.set_title('Integración Numérica', fontsize=14, fontweight='bold')
    
    def _setup_fallback_ui(self):
        """Configura UI alternativa cuando matplotlib no está disponible."""
        self.viz_frame = ttk.LabelFrame(self.parent, text="Visualización", 
                                      padding="10")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        fallback_label = ttk.Label(
            self.viz_frame,
            text="Matplotlib no disponible.\nInstalar con: pip install matplotlib",
            font=('Arial', 12),
            foreground='gray'
        )
        fallback_label.pack(expand=True)
    
    def plot_method(self, func: Callable[[float], float], a: float, b: float, 
                   result: IntegrationResult, method_name: str):
        """
        Grafica la función y el método de integración.
        
        Args:
            func: Función a integrar
            a, b: Límites de integración
            result: Resultado de la integración
            method_name: Nombre del método usado
        """
        if not MATPLOTLIB_AVAILABLE:
            return
        
        self.ax.clear()
        
        # Generar puntos para graficar la función
        x_smooth = [a + i * (b - a) / 200 for i in range(201)]
        try:
            y_smooth = [func(x) for x in x_smooth]
        except:
            # Si hay error, usar menos puntos
            x_smooth = [a + i * (b - a) / 50 for i in range(51)]
            y_smooth = []
            for x in x_smooth:
                try:
                    y_smooth.append(func(x))
                except:
                    y_smooth.append(0)  # Valor por defecto
        
        # Graficar función
        self.ax.plot(x_smooth, y_smooth, 'b-', linewidth=2, label=f'f(x) = {result.method_name}')
        
        # Graficar método específico
        if method_name == 'rectangulo':
            self._plot_rectangle_method(func, a, b, result)
        elif method_name == 'trapezoidal':
            self._plot_trapezoidal_method(func, a, b, result)
        elif method_name in ['simpson_13', 'simpson_38']:
            self._plot_simpson_method(func, a, b, result, method_name)
        elif method_name == 'boole':
            self._plot_boole_method(func, a, b, result)
        elif method_name == 'adaptativo':
            self._plot_adaptive_method(func, a, b, result)
        
        # Configurar gráfica
        self.ax.set_xlabel('x', fontsize=12)
        self.ax.set_ylabel('f(x)', fontsize=12)
        self.ax.set_title(f'Integración Numérica - {result.method_name}', 
                         fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Actualizar canvas
        self.canvas.draw()
    
    def _plot_rectangle_method(self, func: Callable, a: float, b: float, 
                             result: IntegrationResult):
        """Visualiza el método del rectángulo."""
        n = len(result.points)
        h = result.step_size
        
        for i, point in enumerate(result.points):
            # Rectángulo desde x_i hasta x_{i+1}
            x_left = a + i * h
            x_right = a + (i + 1) * h
            
            # Altura en el punto medio
            height = point.fx
            
            # Dibujar rectángulo
            rect_x = [x_left, x_left, x_right, x_right, x_left]
            rect_y = [0, height, height, 0, 0]
            
            self.ax.fill(rect_x, rect_y, color='cyan', alpha=0.3, 
                        edgecolor='blue', linewidth=1)
            
            # Marcar punto medio
            self.ax.plot(point.x, height, 'ro', markersize=4)
        
        self.ax.plot([], [], color='cyan', alpha=0.3, label='Rectángulos')
    
    def _plot_trapezoidal_method(self, func: Callable, a: float, b: float, 
                               result: IntegrationResult):
        """Visualiza el método trapezoidal."""
        # Extraer puntos x e y
        x_points = [point.x for point in result.points]
        y_points = [point.fx for point in result.points]
        
        # Crear trapecios
        for i in range(len(x_points) - 1):
            x_trap = [x_points[i], x_points[i], x_points[i+1], x_points[i+1]]
            y_trap = [0, y_points[i], y_points[i+1], 0]
            
            self.ax.fill(x_trap, y_trap, color='lightcoral', alpha=0.4,
                        edgecolor='red', linewidth=1)
        
        # Marcar puntos de evaluación
        self.ax.plot(x_points, y_points, 'ro-', markersize=5, linewidth=2,
                    label='Puntos de evaluación')
    
    def _plot_simpson_method(self, func: Callable, a: float, b: float, 
                           result: IntegrationResult, method_name: str):
        """Visualiza métodos de Simpson."""
        x_points = [point.x for point in result.points]
        y_points = [point.fx for point in result.points]
        
        # Color según método
        color = 'lightgreen' if method_name == 'simpson_13' else 'peachpuff'
        edge_color = 'green' if method_name == 'simpson_13' else 'orange'
        
        # Para Simpson, aproximar con parábolas (simplificado)
        step = 3 if method_name == 'simpson_38' else 2
        
        for i in range(0, len(x_points) - step, step):
            x_seg = x_points[i:i+step+1]
            y_seg = y_points[i:i+step+1]
            
            # Llenar área bajo los puntos (aproximación)
            x_fill = x_seg + [x_seg[0]]
            y_fill = y_seg + [0]
            
            self.ax.fill(x_fill, y_fill, color=color, alpha=0.4,
                        edgecolor=edge_color, linewidth=1)
        
        # Marcar puntos
        self.ax.plot(x_points, y_points, 'go-', markersize=5, 
                    label=f'Puntos {method_name.replace("_", " ").title()}')
    
    def _plot_boole_method(self, func: Callable, a: float, b: float, 
                         result: IntegrationResult):
        """Visualiza el método de Boole."""
        x_points = [point.x for point in result.points]
        y_points = [point.fx for point in result.points]
        
        # Agrupar en segmentos de 5 puntos
        for i in range(0, len(x_points) - 4, 4):
            x_seg = x_points[i:i+5]
            y_seg = y_points[i:i+5]
            
            # Llenar área (aproximación)
            x_fill = x_seg + [x_seg[0]]
            y_fill = y_seg + [0]
            
            self.ax.fill(x_fill, y_fill, color='plum', alpha=0.4,
                        edgecolor='purple', linewidth=1)
        
        self.ax.plot(x_points, y_points, 'mo-', markersize=5, 
                    label='Puntos Boole')
    
    def _plot_adaptive_method(self, func: Callable, a: float, b: float, 
                            result: IntegrationResult):
        """Visualiza el método adaptativo."""
        # Para adaptativo, mostrar los puntos evaluados
        if result.points:
            x_points = [point.x for point in result.points]
            y_points = [point.fx for point in result.points]
            
            self.ax.scatter(x_points, y_points, color='red', s=20, 
                          alpha=0.7, label='Puntos evaluados')
        
        # Sombreado simple del área
        x_fill = [a, b, b, a]
        y_min = min([func(x) for x in [a, (a+b)/2, b]])
        y_max = max([func(x) for x in [a, (a+b)/2, b]]) 
        y_fill = [0, 0, y_max, y_min] if y_min >= 0 else [y_min, y_max, y_max, y_min]
        
        self.ax.fill(x_fill, y_fill, color='lightblue', alpha=0.3,
                    label='Área aproximada')
    
    def clear(self):
        """Limpia la visualización."""
        if MATPLOTLIB_AVAILABLE and self.ax:
            self.ax.clear()
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('x', fontsize=12)
            self.ax.set_ylabel('f(x)', fontsize=12)
            self.ax.set_title('Integración Numérica', fontsize=14, fontweight='bold')
            self.canvas.draw()
    
    def save_plot(self, filename: str = "integracion_plot.png"):
        """
        Guarda la gráfica actual.
        
        Args:
            filename: Nombre del archivo de salida
        """
        if MATPLOTLIB_AVAILABLE and self.fig:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                return True
            except Exception as e:
                print(f"Error guardando gráfica: {e}")
                return False
        return False
