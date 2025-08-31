#!/usr/bin/env python3
"""
Lanzador simple para el Simulador de Integración Numérica
Ejecuta directamente la interfaz gráfica.
"""

import sys
import os

# Agregar el directorio del proyecto al path
repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_path)

try:
    from integracion_numerica.gui.main_window import MainWindow
    
    if __name__ == "__main__":
        print("🚀 Iniciando Simulador de Integración Numérica...")
        print("   Interfaz Gráfica Mejorada con Scroll")
        print("-" * 50)
        
        # Crear y ejecutar la aplicación
        app = MainWindow()
        app.run()
        
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    print("Asegúrate de estar en el directorio correcto del proyecto.")
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    print("Verifica que todos los archivos estén en su lugar.")
