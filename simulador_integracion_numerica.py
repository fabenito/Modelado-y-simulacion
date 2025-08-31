#!/usr/bin/env python3
"""
Simulador de Integración Numérica
=================================

Simulador completo para métodos de integración numérica Newton-Cotes.
Incluye interfaz gráfica mejorada con scroll, visualización y cálculos detallados.

Uso:
    python simulador_integracion_numerica.py

Métodos implementados:
    - Rectángulo (Punto medio)
    - Trapezoidal
    - Simpson 1/3
    - Simpson 3/8
    - Boole
    - Adaptativo Simpson

Arquitectura:
    - Modular con separación de responsabilidades
    - GUI responsiva con scroll
    - Visualización mejorada
    - Código limpio y mantenible
"""

import sys
import os
from tkinter import messagebox

# Agregar el directorio del proyecto al path
repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_path)

def main():
    """Punto de entrada principal del simulador."""
    print("Simulador de Integración Numérica")
    print("=" * 50)
    print("Métodos disponibles:")
    print("  • Rectángulo (Punto medio)")
    print("  • Trapezoidal") 
    print("  • Simpson 1/3")
    print("  • Simpson 3/8")
    print("  • Boole")
    print("  • Adaptativo Simpson")
    print("=" * 50)
    
    try:
        from integracion_numerica.gui.main_window import MainWindow
        print("Iniciando interfaz gráfica ...")
        
        app = MainWindow()
        app.run()
        
    except ImportError as e:
        error_msg = (
            "Error: No se pudo importar el módulo de integración.\n\n"
            "Verifica que el directorio 'integracion_numerica/' existe\n"
            "y contiene todos los módulos necesarios.\n\n"
            f"Error técnico: {e}"
        )
        print(error_msg)
        messagebox.showerror("Error de Importación", error_msg)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f" Error inesperado al iniciar el simulador: {e}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simulador de Integración Numérica
=================================

Simulador completo para métodos de integración numérica Newton-Cotes.
Incluye interfaz gráfica mejorada con scroll, visualización y cálculos detallados.

Uso:
    python simulador_integracion_numerica.py

Métodos implementados:
    - Rectángulo (Punto medio)
    - Trapezoidal
    - Simpson 1/3
    - Simpson 3/8
    - Boole
    - Adaptativo Simpson

Arquitectura:
    - Modular con separación de responsabilidades
    - GUI responsiva con scroll
    - Visualización mejorada
    - Código limpio y mantenible
"""

import sys
import os
from tkinter import messagebox

# Agregar el directorio del proyecto al path
repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_path)

def main():
    """Punto de entrada principal del simulador."""
    print("🧮 Simulador de Integración Numérica")
    print("=" * 50)
    print("Métodos disponibles:")
    print("  • Rectángulo (Punto medio)")
    print("  • Trapezoidal") 
    print("  • Simpson 1/3")
    print("  • Simpson 3/8")
    print("  • Boole")
    print("  • Adaptativo Simpson")
    print("=" * 50)
    
    try:
        from integracion_numerica.gui.main_window import MainWindow
        print("🚀 Iniciando interfaz gráfica modular...")
        
        app = MainWindow()
        app.run()
        
    except ImportError as e:
        error_msg = (
            "❌ Error: No se pudo importar el módulo de integración.\n\n"
            "Verifica que el directorio 'integracion_numerica/' existe\n"
            "y contiene todos los módulos necesarios.\n\n"
            f"Error técnico: {e}"
        )
        print(error_msg)
        messagebox.showerror("Error de Importación", error_msg)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"❌ Error inesperado al iniciar el simulador: {e}"
        print(error_msg)
        messagebox.showerror("Error", error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
