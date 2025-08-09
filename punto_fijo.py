import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

def fixed_point_iteration(g, x0, tol=1e-5, max_iter=100):
    """
    Método de punto fijo simple
    g(x): función de iteración
    x0: valor inicial
    """
    x = x0
    iter_values = [x0]
    
    for i in range(max_iter):
        x_new = g(x)
        iter_values.append(x_new)
        
        if abs(x_new - x) < tol:
            print("Tolerancia alcanzada...")
            break
            
        x = x_new
        
    return x_new, iter_values

def plot_iteration(g, iter_values, a, b):
    """
    Graficar el proceso iterativo
    """
    x = np.linspace(a, b, 1000)
    y = [g(xi) for xi in x]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', label='g(x)')
    plt.plot(x, x, 'k--', label='y = x')
    plt.plot(iter_values[-1], iter_values[-1], 'r*', markersize=10, label='Punto Fijo')
    
    # Graficar iteraciones
    for i in range(len(iter_values)-1):
        x1, y1 = iter_values[i], iter_values[i]
        x2, y2 = x1, g(x1)
        x3, y3 = iter_values[i+1], y2
        plt.plot([x1, x2], [y1, y2], 'g:', alpha=0.3)
        plt.plot([x2, x3], [y2, y3], 'g:', alpha=0.3)
    
    plt.grid(True)
    plt.legend()
    plt.title('Método del Punto Fijo')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

class FixedPointGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Método del Punto Fijo Simple")
        self.create_widgets()
    
    def create_widgets(self):
        # Modo de entrada
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=5)
        
        self.mode_var = tk.StringVar(value="g")
        tk.Radiobutton(mode_frame, text="Ingresar g(x)", variable=self.mode_var, 
                      value="g", command=self.update_example).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Ingresar f(x)", variable=self.mode_var, 
                      value="f", command=self.update_example).pack(side=tk.LEFT)
        
        # Función
        tk.Label(self.root, text="Función (usar x como variable):").pack(pady=5)
        self.func_entry = tk.Entry(self.root, width=40)
        self.func_entry.pack(pady=5)
        
        # Valor inicial
        initial_frame = tk.Frame(self.root)
        initial_frame.pack(pady=5)
        tk.Label(initial_frame, text="x₀:").pack(side=tk.LEFT)
        self.x0_entry = tk.Entry(initial_frame, width=10)
        self.x0_entry.insert(0, "1.0")
        self.x0_entry.pack(side=tk.LEFT, padx=5)
        
        # Rango para graficar
        range_frame = tk.Frame(self.root)
        range_frame.pack(pady=5)
        tk.Label(range_frame, text="Rango: [").pack(side=tk.LEFT)
        self.a_entry = tk.Entry(range_frame, width=8)
        self.a_entry.insert(0, "-2")
        self.a_entry.pack(side=tk.LEFT)
        tk.Label(range_frame, text=", ").pack(side=tk.LEFT)
        self.b_entry = tk.Entry(range_frame, width=8)
        self.b_entry.insert(0, "2")
        self.b_entry.pack(side=tk.LEFT)
        tk.Label(range_frame, text="]").pack(side=tk.LEFT)
        
        # Botón calcular
        tk.Button(self.root, text="Calcular", command=self.calculate).pack(pady=10)
        
        # Resultados
        self.result_text = tk.Text(self.root, height=4, width=40)
        self.result_text.pack(pady=5)
        
        self.update_example()
    
    def update_example(self):
        if self.mode_var.get() == "g":
            self.func_entry.delete(0, tk.END)
            self.func_entry.insert(0, "cos(x) + x")
        else:
            self.func_entry.delete(0, tk.END)
            self.func_entry.insert(0, "cos(x)")
    
    def calculate(self):
        try:
            # Crear espacio de nombres seguro
            namespace = {
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
                'pi': np.pi, 'e': np.e
            }
            
            # Obtener función
            func_str = self.func_entry.get()
            if self.mode_var.get() == "f":
                f = lambda x: eval(func_str, namespace | {'x': x})
                g = lambda x: x + f(x)  # Convertir f(x) = 0 a x = g(x)
            else:
                g = lambda x: eval(func_str, namespace | {'x': x})
            
            # Obtener parámetros
            x0 = float(self.x0_entry.get())
            a = float(self.a_entry.get())
            b = float(self.b_entry.get())
            
            # Calcular
            root, iter_values = fixed_point_iteration(g, x0)
            
            # Mostrar resultados
            self.result_text.delete(1.0, tk.END)
            if self.mode_var.get() == "f":
                self.result_text.insert(tk.END, 
                    f"Raíz encontrada: {root:.8f}\n"
                    f"f(x) en la raíz: {float(f(root)):.2e}\n"
                    f"Iteraciones: {len(iter_values)-1}")
            else:
                self.result_text.insert(tk.END, 
                    f"Punto fijo encontrado: {root:.8f}\n"
                    f"g(x) - x en punto fijo: {float(g(root) - root):.2e}\n"
                    f"Iteraciones: {len(iter_values)-1}")
            
            # Graficar
            plot_iteration(g, iter_values, a, b)
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Error: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FixedPointGUI(root)
    root.mainloop()
