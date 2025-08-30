"""
Componente para mostrar resultados de integración numérica.
Incluye tablas detalladas y resumen de resultados.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
from ..methods.base import IntegrationResult
from ..utils import format_result


class ResultDisplay:
    """
    Componente para mostrar resultados de integración en tabla.
    """
    
    def __init__(self, parent: tk.Widget):
        """
        Inicializa el display de resultados.
        
        Args:
            parent: Widget padre donde crear la tabla
        """
        self.parent = parent
        self._setup_ui()
    
    def _setup_ui(self):
        """Configura la interfaz de la tabla de resultados."""
        # Marco principal para resultados
        self.result_frame = ttk.LabelFrame(self.parent, text="Resultados Detallados", 
                                         padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Crear Treeview para mostrar tabla
        columns = ("i", "x_i", "f(x_i)", "coef", "contribución")
        self.tree = ttk.Treeview(self.result_frame, columns=columns, show="headings", 
                               height=10)
        
        # Configurar headers
        self.tree.heading("i", text="i")
        self.tree.heading("x_i", text="x_i") 
        self.tree.heading("f(x_i)", text="f(x_i)")
        self.tree.heading("coef", text="coef")
        self.tree.heading("contribución", text="contribución")
        
        # Configurar anchos de columna
        self.tree.column("i", width=50, anchor="center")
        self.tree.column("x_i", width=120, anchor="center")
        self.tree.column("f(x_i)", width=120, anchor="center")
        self.tree.column("coef", width=80, anchor="center")
        self.tree.column("contribución", width=120, anchor="center")
        
        # Scrollbar para la tabla
        scrollbar = ttk.Scrollbar(self.result_frame, orient="vertical", 
                                command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tabla y scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Frame para resumen de resultados
        self.summary_frame = ttk.Frame(self.parent)
        self.summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Labels para mostrar resultado final
        self.result_label = ttk.Label(self.summary_frame, text="", 
                                    font=('Arial', 12, 'bold'),
                                    foreground='darkblue')
        self.result_label.pack(side=tk.LEFT, padx=10)
        
        self.info_label = ttk.Label(self.summary_frame, text="", 
                                  font=('Arial', 10),
                                  foreground='gray')
        self.info_label.pack(side=tk.LEFT, padx=10)
    
    def show_result(self, result: IntegrationResult, method_name: str):
        """
        Muestra los resultados de integración en la tabla.
        
        Args:
            result: Resultado de la integración
            method_name: Nombre del método usado
        """
        # Limpiar tabla anterior
        self.clear()
        
        # Llenar tabla con puntos de integración
        for point in result.points:
            values = (
                str(point.index),
                format_result(point.x, 10),
                format_result(point.fx, 10),
                format_result(point.coefficient, 3),
                format_result(point.contribution, 12)
            )
            self.tree.insert("", tk.END, values=values)
        
        # Actualizar resultado final
        result_text = f"Resultado: {format_result(result.value, 12)}"
        self.result_label.config(text=result_text)
        
        # Información adicional
        info_parts = [
            f"Método: {result.method_name}",
            f"h = {format_result(result.step_size, 8)}",
            f"Evaluaciones: {result.function_evaluations}"
        ]
        
        if result.error_estimate is not None:
            info_parts.append(f"Error est.: {format_result(result.error_estimate, 6)}")
        
        info_text = " | ".join(info_parts)
        self.info_label.config(text=info_text)
        
        # Actualizar título del frame
        self.result_frame.config(text=f"Resultados Detallados - {result.method_name}")
    
    def show_adaptive_result(self, result: IntegrationResult, history: list):
        """
        Muestra resultados específicos para método adaptativo.
        
        Args:
            result: Resultado de integración adaptativa
            history: Historia del proceso adaptativo
        """
        # Limpiar tabla
        self.clear()
        
        # Cambiar headers para método adaptativo
        headers = ("Profundidad", "x_medio", "f(x)", "Error Est.", "Contribución")
        for i, header in enumerate(headers):
            self.tree.heading(f"#{i+1}", text=header)
        
        # Llenar con datos adaptativos
        for i, entry in enumerate(history):
            if isinstance(entry, dict):
                values = (
                    str(entry.get('depth', i)),
                    format_result(entry.get('x', 0), 10),
                    format_result(entry.get('fx', 0), 10),
                    format_result(entry.get('error_estimate', 0), 8),
                    format_result(entry.get('interval_contribution', 0), 10)
                )
                self.tree.insert("", tk.END, values=values)
        
        # Mostrar resultado final
        result_text = f"Resultado: {format_result(result.value, 12)}"
        self.result_label.config(text=result_text)
        
        info_text = (f"Método: {result.method_name} | "
                    f"Tolerancia: {format_result(result.error_estimate or 0, 6)} | "
                    f"Evaluaciones: {result.function_evaluations}")
        self.info_label.config(text=info_text)
    
    def clear(self):
        """Limpia la tabla y etiquetas de resultados."""
        # Limpiar tabla
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Restaurar headers originales
        headers = ("i", "x_i", "f(x_i)", "coef", "contribución")
        for i, header in enumerate(headers):
            self.tree.heading(f"#{i+1}", text=header)
        
        # Limpiar labels
        self.result_label.config(text="")
        self.info_label.config(text="")
        
        # Restaurar título
        self.result_frame.config(text="Resultados Detallados")
    
    def export_results(self, filename: str = "resultados_integracion.csv"):
        """
        Exporta los resultados a un archivo CSV.
        
        Args:
            filename: Nombre del archivo de salida
        """
        try:
            import csv
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Escribir headers
                headers = [self.tree.heading(col)['text'] for col in self.tree['columns']]
                writer.writerow(headers)
                
                # Escribir datos
                for item in self.tree.get_children():
                    values = self.tree.item(item)['values']
                    writer.writerow(values)
                
                # Escribir resumen
                writer.writerow([])
                writer.writerow(['Resumen'])
                writer.writerow(['Resultado:', self.result_label.cget('text')])
                writer.writerow(['Información:', self.info_label.cget('text')])
            
            return True
            
        except Exception as e:
            print(f"Error exportando resultados: {e}")
            return False
