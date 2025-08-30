#!/usr/bin/env python3
"""
Simulador de Integración Numérica - Versión Refactorizada
Implementación modular de métodos de Newton-Cotes y técnicas adaptativas.

Autor: Modelado y Simulación UADE
Versión: 2.0.0
"""

import sys
import os

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integracion_numerica.gui import MainWindow


def main():
    """Función principal del simulador."""
    try:
        # Crear y ejecutar la aplicación
        app = MainWindow()
        app.run()
        
    except ImportError as e:
        print(f"Error de importación: {e}")
        print("\nDependencias requeridas:")
        print("- tkinter (incluido en Python estándar)")
        print("- matplotlib (opcional, para visualizaciones): pip install matplotlib")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
