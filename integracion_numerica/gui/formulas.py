"""
Componente para mostrar fórmulas de integración numérica.
Ventana de referencia con fórmulas matemáticas y información teórica.
"""

import tkinter as tk
from tkinter import ttk, messagebox


class FormulaDisplay:
    """
    Ventana que muestra las fórmulas de integración numérica.
    Incluye LaTeX rendering cuando matplotlib está disponible.
    """
    
    def __init__(self, parent: tk.Widget):
        """
        Inicializa la ventana de fórmulas.
        
        Args:
            parent: Widget padre
        """
        self.parent = parent
        self.window = None
        self._create_formula_window()
    
    def _create_formula_window(self):
        """Crea la ventana de fórmulas."""
        self.window = tk.Toplevel(self.parent)
        self.window.title("Fórmulas de Integración Numérica - Newton-Cotes")
        self.window.geometry("1000x750")
        self.window.configure(bg='white')
        
        # Intentar mostrar fórmulas con LaTeX
        try:
            self._create_matplotlib_formulas()
        except:
            self._create_text_formulas()
    
    def _create_matplotlib_formulas(self):
        """Crea fórmulas con renderizado LaTeX usando matplotlib."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ImportError:
            raise ImportError("Matplotlib no disponible")
        
        # Crear figura para las fórmulas
        fig = Figure(figsize=(14, 16), facecolor='white')
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 15)
        ax.axis('off')
        
        # Título principal
        ax.text(5, 14.2, 'Fórmulas de Integración Numérica (Newton-Cotes)', 
               fontsize=20, fontweight='bold', ha='center',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        # Información general
        ax.text(5, 13.6, r'Donde: $h = \frac{b-a}{n}$, $n$ = subdivisiones, $\xi \in [a,b]$', 
               fontsize=12, ha='center', style='italic', color='gray')
        
        # Configurar fórmulas con espaciado
        y_start = 12.8
        y_spacing = 2.8
        
        self._add_rectangle_formula(ax, y_start)
        self._add_trapezoidal_formula(ax, y_start - y_spacing)
        self._add_simpson13_formula(ax, y_start - 2 * y_spacing)
        self._add_simpson38_formula(ax, y_start - 3 * y_spacing)
        self._add_boole_formula(ax, y_start - 4 * y_spacing)
        self._add_adaptive_formula(ax, y_start - 5 * y_spacing)
        
        # Nota final
        ax.text(5, 1.6, '💡 Principio General de Newton-Cotes', 
               fontsize=14, fontweight='bold', ha='center', color='navy')
        ax.text(5, 1.2, 'A mayor grado del polinomio interpolante, mayor precisión', 
               fontsize=12, ha='center', style='italic', color='navy')
        ax.text(5, 0.8, 'Pero también mayor costo computacional y sensibilidad numérica', 
               fontsize=12, ha='center', style='italic', color='navy')
        
        plt.tight_layout()
        
        # Frame principal scrollable
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Canvas y scrollbar
        canvas_widget = tk.Canvas(main_frame, bg='white')
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas_widget.yview)
        scrollable_frame = ttk.Frame(canvas_widget)
        
        # Configurar scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_widget.configure(scrollregion=canvas_widget.bbox("all"))
        )
        
        canvas_widget.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_widget.configure(yscrollcommand=scrollbar.set)
        
        # Integrar matplotlib
        matplotlib_canvas = FigureCanvasTkAgg(fig, scrollable_frame)
        matplotlib_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Pack scrollable components
        canvas_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Botones
        button_frame = ttk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=15, pady=10)
        
        ttk.Button(button_frame, text="Cerrar", 
                  command=self.window.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Guardar", 
                  command=lambda: self._save_formulas(fig)).pack(side=tk.LEFT, padx=5)
    
    def _add_rectangle_formula(self, ax, y_pos):
        """Agrega fórmula del método rectángulo."""
        ax.text(0.5, y_pos, '1. Regla del Rectángulo/Punto Medio (Grado 0)', 
               fontsize=16, fontweight='bold', color='darkblue')
        
        ax.text(1, y_pos - 0.6, r'$I \approx h \sum_{i=0}^{n-1} f\left(\frac{x_i + x_{i+1}}{2}\right)$', 
               fontsize=14, bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcyan', alpha=0.8))
        
        ax.text(1, y_pos - 1.1, r'Error: $E = \frac{(b-a)^3}{24n^2}f^{(2)}(\xi)$', 
               fontsize=12, color='darkblue')
        
        ax.plot([0.2, 9.8], [y_pos - 1.8, y_pos - 1.8], 'k-', alpha=0.2, linewidth=1)
    
    def _add_trapezoidal_formula(self, ax, y_pos):
        """Agrega fórmula del método trapezoidal."""
        ax.text(0.5, y_pos, '2. Regla Trapezoidal (Grado 1)', 
               fontsize=16, fontweight='bold', color='darkred')
        
        ax.text(1, y_pos - 0.6, r'$I \approx \frac{h}{2}[f(a) + f(b)]$', 
               fontsize=14, bbox=dict(boxstyle="round,pad=0.4", facecolor='mistyrose', alpha=0.8))
        
        ax.text(1, y_pos - 1.1, r'Error: $E = -\frac{(b-a)^3}{12n^2}f^{(2)}(\xi)$', 
               fontsize=12, color='darkred')
        
        ax.plot([0.2, 9.8], [y_pos - 1.8, y_pos - 1.8], 'k-', alpha=0.2, linewidth=1)
    
    def _add_simpson13_formula(self, ax, y_pos):
        """Agrega fórmula de Simpson 1/3."""
        ax.text(0.5, y_pos, '3. Regla de Simpson 1/3 (Grado 2)', 
               fontsize=16, fontweight='bold', color='darkgreen')
        
        ax.text(1, y_pos - 0.6, r'$I \approx \frac{h}{3}[f(a) + 4f\left(\frac{a+b}{2}\right) + f(b)]$', 
               fontsize=14, bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
        
        ax.text(1, y_pos - 1.1, r'Error: $E = -\frac{(b-a)^5}{180n^4}f^{(4)}(\xi)$', 
               fontsize=12, color='darkgreen')
        
        ax.plot([0.2, 9.8], [y_pos - 1.8, y_pos - 1.8], 'k-', alpha=0.2, linewidth=1)
    
    def _add_simpson38_formula(self, ax, y_pos):
        """Agrega fórmula de Simpson 3/8."""
        ax.text(0.5, y_pos, '4. Regla de Simpson 3/8 (Grado 3)', 
               fontsize=16, fontweight='bold', color='darkorange')
        
        ax.text(1, y_pos - 0.6, r'$I \approx \frac{3h}{8}[f(x_0) + 3f(x_1) + 3f(x_2) + f(x_3)]$', 
               fontsize=14, bbox=dict(boxstyle="round,pad=0.4", facecolor='peachpuff', alpha=0.8))
        
        ax.text(1, y_pos - 1.1, r'Error: $E = -\frac{3(b-a)^5}{80n^4}f^{(4)}(\xi)$', 
               fontsize=12, color='darkorange')
        
        ax.plot([0.2, 9.8], [y_pos - 1.8, y_pos - 1.8], 'k-', alpha=0.2, linewidth=1)
    
    def _add_boole_formula(self, ax, y_pos):
        """Agrega fórmula de Boole."""
        ax.text(0.5, y_pos, '5. Regla de Boole (Grado 4)', 
               fontsize=16, fontweight='bold', color='purple')
        
        ax.text(1, y_pos - 0.6, r'$I \approx \frac{2h}{45}[7f(x_0) + 32f(x_1) + 12f(x_2) + 32f(x_3) + 7f(x_4)]$', 
               fontsize=14, bbox=dict(boxstyle="round,pad=0.4", facecolor='lavender', alpha=0.8))
        
        ax.text(1, y_pos - 1.1, r'Coeficientes: $\{7, 32, 12, 32, 14, 32, 12, 32, 7\}$ (patrón)', 
               fontsize=12, color='purple')
        
        ax.text(1, y_pos - 1.5, r'Error: $E = -\frac{8(b-a)^7}{945n^6}f^{(6)}(\xi)$', 
               fontsize=12, color='purple')
        
        ax.plot([0.2, 9.8], [y_pos - 2.1, y_pos - 2.1], 'k-', alpha=0.2, linewidth=1)
    
    def _add_adaptive_formula(self, ax, y_pos):
        """Agrega fórmula del método adaptativo."""
        ax.text(0.5, y_pos, '6. Método Adaptativo (Simpson Recursivo)', 
               fontsize=16, fontweight='bold', color='darkblue')
        
        ax.text(1, y_pos - 0.6, r'Estimación de error: $E_{est} = \frac{|S_h - S_{2h}|}{15}$', 
               fontsize=14, bbox=dict(boxstyle="round,pad=0.4", facecolor='lightcyan', alpha=0.8))
        
        ax.text(1, y_pos - 1.1, r'Si $E_{est} < \epsilon$ → aceptar; sino → dividir intervalo', 
               fontsize=12, color='darkblue')
    
    def _create_text_formulas(self):
        """Crea fórmulas como texto plano cuando LaTeX no está disponible."""
        # Frame principal scrollable
        text_frame = ttk.Frame(self.window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Text widget con scrollbar
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Consolas', 11), 
                             bg='white', fg='black')
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Contenido de texto
        formula_text = self._get_text_formulas()
        text_widget.insert(tk.END, formula_text)
        text_widget.config(state=tk.DISABLED)
        
        # Botón para cerrar
        ttk.Button(self.window, text="Cerrar", 
                  command=self.window.destroy).pack(pady=10)
    
    def _get_text_formulas(self) -> str:
        """Retorna las fórmulas como texto plano."""
        return """
═══════════════════════════════════════════════════════════════════════════════
                    FÓRMULAS DE INTEGRACIÓN NUMÉRICA (NEWTON-COTES)
═══════════════════════════════════════════════════════════════════════════════

Donde: h = (b-a)/n,  n = número de subdivisiones,  ξ ∈ [a,b]


───────────────────────────────────────────────────────────────────────────────
1. REGLA DEL RECTÁNGULO/PUNTO MEDIO (Grado 0)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:  I ≈ h∑f((xi + xi+1)/2)  (evaluación en punto medio)
   
   Error:    E = (b-a)³f''(ξ)/(24n²)


───────────────────────────────────────────────────────────────────────────────
2. REGLA TRAPEZOIDAL (Grado 1)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:  I ≈ (h/2)[f(a) + f(b)]
   
   Error:    E = -(b-a)³f''(ξ)/(12n²)


───────────────────────────────────────────────────────────────────────────────
3. REGLA DE SIMPSON 1/3 (Grado 2)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:  I ≈ (h/3)[f(a) + 4f((a+b)/2) + f(b)]
   
   Error:    E = -(b-a)⁵f⁽⁴⁾(ξ)/(180n⁴)


───────────────────────────────────────────────────────────────────────────────
4. REGLA DE SIMPSON 3/8 (Grado 3)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:  I ≈ (3h/8)[f(x₀) + 3f(x₁) + 3f(x₂) + f(x₃)]
   
   Error:    E = -3(b-a)⁵f⁽⁴⁾(ξ)/(80n⁴)


───────────────────────────────────────────────────────────────────────────────
5. REGLA DE BOOLE (Grado 4)
───────────────────────────────────────────────────────────────────────────────
   
   Fórmula:      I ≈ (2h/45)[7f(x₀) + 32f(x₁) + 12f(x₂) + 32f(x₃) + 7f(x₄)]
   
   Coeficientes: {7, 32, 12, 32, 14, 32, 12, 32, 7} (patrón repetitivo)
   
   Error:        E = -8(b-a)⁷f⁽⁶⁾(ξ)/(945n⁶)


───────────────────────────────────────────────────────────────────────────────
6. MÉTODO ADAPTATIVO (Simpson Recursivo)
───────────────────────────────────────────────────────────────────────────────
   
   Estimación de error:  E_est = |S_h - S_2h|/15
   
   Criterio:            Si E_est < ε → aceptar; sino → dividir intervalo
   

═══════════════════════════════════════════════════════════════════════════════
💡 PRINCIPIO GENERAL:
   A mayor grado del polinomio interpolante → Mayor precisión
   Pero también → Mayor costo computacional y sensibilidad numérica
═══════════════════════════════════════════════════════════════════════════════

NOTAS IMPORTANTES:
• El método del Rectángulo es el más simple pero menos preciso
• Simpson 1/3 requiere n par, Simpson 3/8 requiere n múltiplo de 3
• Boole requiere n múltiplo de 4
• El método Adaptativo ajusta automáticamente la precisión
• Para funciones suaves, métodos de mayor grado son más eficientes
• Para funciones irregulares, el método adaptativo es recomendable
"""
    
    def _save_formulas(self, fig):
        """Guarda las fórmulas como imagen."""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Éxito", f"Fórmulas guardadas en: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar: {str(e)}")
