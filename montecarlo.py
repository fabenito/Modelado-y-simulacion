
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import norm
import threading

class MonteCarloSimulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Monte Carlo Simulator")

        # --- Main Frame ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Input Frame ---
        input_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        input_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

        # Formula
        ttk.Label(input_frame, text="Formula (use 'x'):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.formula_var = tk.StringVar(value="atan(x)")
        ttk.Entry(input_frame, textvariable=self.formula_var, width=30).grid(row=0, column=1, pady=2)

        # Iterations
        ttk.Label(input_frame, text="Iterations:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.iterations_var = tk.StringVar(value="10000")
        ttk.Entry(input_frame, textvariable=self.iterations_var).grid(row=1, column=1, pady=2)

        # Seed
        ttk.Label(input_frame, text="Seed (for reproducibility):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(input_frame, textvariable=self.seed_var).grid(row=2, column=1, pady=2)

        # Confidence Interval
        ttk.Label(input_frame, text="Confidence Interval (%):").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.ci_var = tk.StringVar(value="95")
        ttk.Entry(input_frame, textvariable=self.ci_var).grid(row=3, column=1, pady=2)
        
        # Integration Range (X)
        ttk.Label(input_frame, text="X-axis range (min, max):").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.x_min_var = tk.StringVar(value="1")
        self.x_max_var = tk.StringVar(value="2")
        x_range_frame = ttk.Frame(input_frame)
        x_range_frame.grid(row=4, column=1, pady=2, sticky=(tk.W, tk.E))
        ttk.Entry(x_range_frame, textvariable=self.x_min_var, width=7).pack(side=tk.LEFT)
        ttk.Label(x_range_frame, text=" to ").pack(side=tk.LEFT)
        ttk.Entry(x_range_frame, textvariable=self.x_max_var, width=7).pack(side=tk.LEFT)

        # Y-axis max for random points
        ttk.Label(input_frame, text="Y-axis max (for points):").grid(row=5, column=0, sticky=tk.W, pady=2)
        self.y_max_var = tk.StringVar(value="1.2")
        ttk.Entry(input_frame, textvariable=self.y_max_var).grid(row=5, column=1, pady=2)


        # --- Run Button ---
        self.run_button = ttk.Button(input_frame, text="Run Simulation", command=self.start_simulation)
        self.run_button.grid(row=6, column=0, columnspan=2, pady=10)
        
        # --- Progress Bar ---
        self.progress_bar = ttk.Progressbar(input_frame, orient='horizontal', mode='indeterminate')
        self.progress_bar.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)


        # --- Output Frame ---
        output_frame = ttk.LabelFrame(main_frame, text="Result", padding="10")
        output_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
        self.result_var = tk.StringVar(value="Result will be displayed here.")
        ttk.Label(output_frame, textvariable=self.result_var, wraplength=300).grid(row=0, column=0)

        # --- Plot Frame ---
        plot_frame = ttk.LabelFrame(main_frame, text="Graph", padding="10")
        plot_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.root.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def start_simulation(self):
        self.run_button.config(state=tk.DISABLED)
        self.progress_bar.start()
        
        try:
            # --- Parameter Validation ---
            formula_str = self.formula_var.get()
            iterations = int(self.iterations_var.get())
            seed = int(self.seed_var.get())
            ci_level = float(self.ci_var.get())
            x_min = float(self.x_min_var.get())
            x_max = float(self.x_max_var.get())
            y_max = float(self.y_max_var.get())

            if x_min >= x_max or y_max <= 0 or iterations <= 0 or not (0 < ci_level < 100):
                raise ValueError("Invalid input: Check ranges, iterations, and CI level.")

            # --- Run simulation in a separate thread to keep UI responsive ---
            sim_thread = threading.Thread(
                target=self.run_simulation_task,
                args=(formula_str, iterations, seed, ci_level, x_min, x_max, y_max)
            )
            sim_thread.start()

        except (ValueError, SyntaxError) as e:
            messagebox.showerror("Input Error", f"Please check your inputs. Error: {e}")
            self.run_button.config(state=tk.NORMAL)
            self.progress_bar.stop()

    def run_simulation_task(self, formula_str, iterations, seed, ci_level, x_min, x_max, y_max):
        try:
            # --- Core Monte Carlo Logic ---
            np.random.seed(seed) 
            
            # Generate random points
            rand_x = np.random.uniform(x_min, x_max, iterations)
            rand_y = np.random.uniform(0, y_max, iterations)

            # Evaluate the user's formula
            # Use a restricted namespace for safety
            safe_dict = {
                "np": np,
                "x": rand_x,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'atan': np.arctan,
                'exp': np.exp, 'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10
            }
            func_y = eval(formula_str, {"__builtins__": {}}, safe_dict)

            # Find points under the curve
            points_under = rand_y < func_y
            count_under = np.sum(points_under)

            # Calculate area
            box_area = (x_max - x_min) * y_max
            estimated_area = (count_under / iterations) * box_area
            
            # --- Confidence Interval Calculation ---
            p_hat = count_under / iterations
            z_score = norm.ppf(1 - (1 - ci_level / 100) / 2)
            margin_of_error = z_score * np.sqrt((p_hat * (1 - p_hat)) / iterations)
            ci_lower = (p_hat - margin_of_error) * box_area
            ci_upper = (p_hat + margin_of_error) * box_area

            # --- Schedule UI update on the main thread ---
            self.root.after(0, self.update_ui, estimated_area, ci_lower, ci_upper, ci_level, 
                            rand_x, rand_y, points_under, formula_str, x_min, x_max, y_max)

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Simulation Error", f"An error occurred: {e}"))
        finally:
            self.root.after(0, self.simulation_finished)
            
    def simulation_finished(self):
        self.progress_bar.stop()
        self.run_button.config(state=tk.NORMAL)

    def update_ui(self, area, ci_lower, ci_upper, ci_level, rand_x, rand_y, points_under, formula_str, x_min, x_max, y_max):
        # --- Update Result Text ---
        result_text = (
            f"Estimated Area: {area:.6f}"
            f"{ci_level}% Confidence Interval: [{ci_lower:.6f}, {ci_upper:.6f}]"
        )
        self.result_var.set(result_text)

        # --- Update Plot ---
        self.ax.clear() 
        
        # Plot random points
        self.ax.scatter(rand_x[points_under], rand_y[points_under], color='green', alpha=0.3, s=5, label='Points Under Curve')
        self.ax.scatter(rand_x[~points_under], rand_y[~points_under], color='red', alpha=0.3, s=5, label='Points Over Curve')

        # Plot the function curve
        line_x = np.linspace(x_min, x_max, 200)
        safe_dict_line = {
            "np": np, "x": line_x,
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'atan': np.arctan,
            'exp': np.exp, 'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10
        }
        line_y = eval(formula_str, {"__builtins__": {}}, safe_dict_line)
        self.ax.plot(line_x, line_y, color='blue', linewidth=2, label=f"f(x) = {formula_str}")

        # Style the plot
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(0, y_max)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Monte Carlo Simulation")
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = MonteCarloSimulatorApp(root)
    root.mainloop()
