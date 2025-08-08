import numpy as np
import matplotlib.pyplot as plt

def bolzano_method(f, a, b, max_iterations=100, tolerance=1e-6):
    
    if f(a) * f(b) >= 0:
        raise ValueError("Las funciones deben tener signos opuestos en los extremos del intervalo")
    
    iterations = 0
    history = []
    
    while iterations < max_iterations:
        c = (a + b) / 2
        fc = f(c)
        history.append(c)
        
        # verifico si encontré una raíz o si llegué a la precisión deseada
        if abs(fc) < tolerance:
            return c, iterations, history
        
        # actualizo el intervalo
        if f(a) * fc < 0:
            b = c
        else:
            a = c
            
        iterations += 1
        
        # verifico si el intervalo es suficientemente pequeño
        if abs(b - a) < tolerance:
            return c, iterations, history
    
    return c, iterations, history

def plot_function_and_root(f, a, b, root, num_points=1000):

    x = np.linspace(a, b, num_points)
    y = f(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='f(x)')
    plt.plot(x, np.zeros_like(x), 'k--', alpha=0.3)  # x-axis
    plt.plot(root, 0, 'r*', markersize=10, label='Raiz')
    
    plt.grid(True)
    plt.legend()
    plt.title('Funcion y Raíz Encontrada')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()


class BolzanoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora del Método de Bolzano")
        
        # Creación de widgets
        self.create_widgets()
        
    def create_widgets(self):
        # Ingreso de la función
        tk.Label(self.root, text="Función (usar x como variable, ej., x**2 - 4):").pack(pady=5)
        self.func_entry = tk.Entry(self.root, width=40)
        self.func_entry.insert(0, "x**3 - x - 2")
        self.func_entry.pack(pady=5)
        
        # Ingreso del intervalo
        interval_frame = tk.Frame(self.root)
        interval_frame.pack(pady=5)
        
        tk.Label(interval_frame, text="a:").pack(side=tk.LEFT)
        self.a_entry = tk.Entry(interval_frame, width=10)
        self.a_entry.insert(0, "1")
        self.a_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(interval_frame, text="b:").pack(side=tk.LEFT)
        self.b_entry = tk.Entry(interval_frame, width=10)
        self.b_entry.insert(0, "2")
        self.b_entry.pack(side=tk.LEFT, padx=5)
        
        # Iteraciones máximas y tolerancia
        params_frame = tk.Frame(self.root)
        params_frame.pack(pady=5)
        
        tk.Label(params_frame, text="Iteraciones máximas:").pack(side=tk.LEFT)
        self.iter_entry = tk.Entry(params_frame, width=10)
        self.iter_entry.insert(0, "100")
        self.iter_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(params_frame, text="Tolerancia:").pack(side=tk.LEFT)
        self.tol_entry = tk.Entry(params_frame, width=10)
        self.tol_entry.insert(0, "1e-6")
        self.tol_entry.pack(side=tk.LEFT, padx=5)
        
        # Botón de cálculo
        tk.Button(self.root, text="Calcular Raíz", command=self.calculate).pack(pady=10)
        
        # Mostrar resultados
        self.result_text = tk.Text(self.root, height=4, width=40)
        self.result_text.pack(pady=5)
        
    def calculate(self):
        try:
            # Obtener función desde el string
            func_str = self.func_entry.get()
            func = lambda x: eval(func_str.replace('x', 'x'))
            
            # Obtener parámetros
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            max_iter = int(self.iter_entry.get())
            tol = float(self.tol_entry.get())
            
            # Calcular raíz
            root, iterations, history = bolzano_method(func, a, b, max_iter, tol)
            
            # Mostrar resultados
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, 
                f"Raíz encontrada: {root:.8f}\n"
                f"Valor de la función en la raíz: {func(root):.2e}\n"
                f"Número de iteraciones: {iterations}")
            
            # Graficar la función
            plot_function_and_root(func, a, b, root)
            
        except ValueError as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")

if __name__ == "__main__":
    import tkinter as tk
    root = tk.Tk()
    app = BolzanoGUI(root)
    root.mainloop()
