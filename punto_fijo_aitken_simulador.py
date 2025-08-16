#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador del Método del Punto Fijo con Aceleración de Aitken Δ²
Autor: ChatGPT
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from math import *
import math
import numpy as np
import pandas as pd

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# -------- Evaluación segura de expresiones --------
def build_safe_env(x_value: float):
    safe_env = {
        "x": float(x_value),
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos, "atan": math.atan,
        "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh, "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
        "exp": math.exp, "log": math.log, "log10": math.log10, "sqrt": math.sqrt, "abs": abs, "floor": math.floor, "ceil": math.ceil,
        "pi": math.pi, "e": math.e,
        "np": np,
    }
    return safe_env

def eval_g(expr: str, x_value: float) -> float:
    safe_env = build_safe_env(x_value)
    return float(eval(expr, {"__builtins__": {}}, safe_env))

# --------- Lógica del método de punto fijo ---------
def punto_fijo_aitken(g_expr: str, x0: float, max_iter: int, tol: float, usar_aitken: bool):
    data = []
    x_vals = [x0]
    x_n = x0
    for k in range(1, max_iter + 1):
        gxn = eval_g(g_expr, x_n)
        err_abs = abs(gxn - x_n)
        err_rel = err_abs / (abs(gxn) + 1e-15)
        x_vals.append(gxn)

        xA = None
        if usar_aitken and k >= 2:
            # necesitamos x_{n-2}, x_{n-1}, x_n
            x0a = x_vals[-3]
            x1a = x_vals[-2]
            x2a = x_vals[-1]
            denom = x2a - 2*x1a + x0a
            if abs(denom) > 1e-15:
                xA = x0a - (x1a - x0a)**2 / denom

        data.append((k, x_n, gxn, err_abs, err_rel, xA))

        # Criterio de parada: si err_abs < tol o err_rel < tol o Aitken converge
        if err_abs <= tol or err_rel <= tol:
            return data, gxn, k, True
        if usar_aitken and xA is not None:
            if abs(xA - x_n) <= tol or abs(xA - gxn) <= tol:
                return data, xA, k, True

        x_n = gxn
    return data, x_n, max_iter, False

# --------- Aplicación Tkinter ---------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Simulador - Punto Fijo + Aitken")
        self.geometry("1150x700")
        self.minsize(1050, 650)

        self.g_expr_var = tk.StringVar(value="cos(x)")
        self.x0_var = tk.StringVar(value="0.5")
        self.maxiter_var = tk.StringVar(value="50")
        self.tol_var = tk.StringVar(value="1e-8")
        self.decimales_var = tk.IntVar(value=12)
        self.show_cobweb_var = tk.BooleanVar(value=True)
        self.usar_aitken_var = tk.BooleanVar(value=True)

        self._build_ui()
        self.last_data = None
        self.last_root = None
        self.last_converged = None

    def _build_ui(self):
        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="g(x):").grid(row=0, column=0, sticky="w", padx=(0,4))
        ttk.Entry(top, textvariable=self.g_expr_var, width=38).grid(row=0, column=1, padx=(0,12))

        ttk.Label(top, text="Semilla x₀:").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.x0_var, width=10).grid(row=0, column=3, padx=(4,12))

        ttk.Label(top, text="Iteraciones máx:").grid(row=0, column=4, sticky="w")
        ttk.Entry(top, textvariable=self.maxiter_var, width=8).grid(row=0, column=5, padx=(4,12))

        ttk.Label(top, text="Tolerancia:").grid(row=0, column=6, sticky="w")
        ttk.Entry(top, textvariable=self.tol_var, width=10).grid(row=0, column=7, padx=(4,12))

        ttk.Label(top, text="Decimales:").grid(row=0, column=8, sticky="w")
        ttk.Spinbox(top, from_=4, to=18, textvariable=self.decimales_var, width=5).grid(row=0, column=9, padx=(4,12))

        ttk.Checkbutton(top, text="Graficar telaraña", variable=self.show_cobweb_var).grid(row=0, column=10, padx=4)
        ttk.Checkbutton(top, text="Usar Aitken Δ²", variable=self.usar_aitken_var).grid(row=0, column=11, padx=4)

        btns = ttk.Frame(self, padding=(8,0,8,8))
        btns.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(btns, text="Ejecutar", command=self.on_run).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Graficar", command=self.on_plot).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Guardar CSV", command=self.on_save_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Limpiar", command=self.on_clear).pack(side=tk.LEFT, padx=4)

        main = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, padding=8)
        main.add(left, weight=1)

        cols = ("iter", "xn", "gxn", "err_abs", "err_rel", "xAitken")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=18)
        headers = ["Iteración", "x_n", "g(x_n)", "Error Abs", "Error Rel", "x^(Aitken)"]
        for c, h in zip(cols, headers):
            self.tree.heading(c, text=h)
            self.tree.column(c, anchor=tk.E, width=120)
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.status = ttk.Label(left, text="Listo.", anchor="w")
        self.status.pack(fill=tk.X, pady=(6,0))

        right = ttk.Frame(main, padding=8)
        main.add(right, weight=1)
        self.fig = plt.Figure(figsize=(5,4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, right)
        self.toolbar.update()

    def format_num(self, val):
        if val is None:
            return ""
        dec = int(self.decimales_var.get())
        return f"{float(val):.{dec}f}"

    def read_params(self):
        g_expr = self.g_expr_var.get().strip()
        x0 = float(eval(self.x0_var.get(), {"__builtins__": {}}, {}))
        max_iter = int(eval(self.maxiter_var.get(), {"__builtins__": {}}, {}))
        tol = float(eval(self.tol_var.get(), {"__builtins__": {}}, {}))
        usar_aitken = self.usar_aitken_var.get()
        return g_expr, x0, max_iter, tol, usar_aitken

    def on_clear(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.ax.clear()
        self.canvas.draw()
        self.status.config(text="Listo.")
        self.last_data = None

    def on_run(self):
        try:
            g_expr, x0, max_iter, tol, usar_aitken = self.read_params()
            data, root, k, converged = punto_fijo_aitken(g_expr, x0, max_iter, tol, usar_aitken)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        for row in self.tree.get_children():
            self.tree.delete(row)
        for it, xn, gxn, ea, er, xA in data:
            self.tree.insert("", tk.END, values=(it, self.format_num(xn), self.format_num(gxn),
                                                 self.format_num(ea), self.format_num(er), self.format_num(xA)))
        self.last_data = data
        self.last_root = root
        self.last_converged = converged
        if converged:
            self.status.config(text=f"Convergió a {self.format_num(root)} en {len(data)} iteraciones.")
        else:
            self.status.config(text=f"No convergió en {len(data)} iteraciones.")

    def on_plot(self):
        if not self.last_data:
            self.on_run()
            if not self.last_data:
                return
        g_expr, x0, max_iter, tol, usar_aitken = self.read_params()
        xs = np.linspace(min([r[1] for r in self.last_data])-1, max([r[1] for r in self.last_data])+1, 400)
        ys = [eval_g(g_expr, xv) for xv in xs]
        self.ax.clear()
        self.ax.plot(xs, ys, label="g(x)")
        self.ax.plot(xs, xs, '--', label="y=x")
        if usar_aitken:
            aitken_vals = [r[5] for r in self.last_data if r[5] is not None]
            if aitken_vals:
                self.ax.scatter(aitken_vals, aitken_vals, color="red", label="Aitken")
        if self.show_cobweb_var.get():
            for it, xn, gxn, ea, er, xA in self.last_data:
                self.ax.plot([xn, xn], [xn, gxn], 'k-', alpha=0.5)
                self.ax.plot([xn, gxn], [gxn, gxn], 'k-', alpha=0.5)
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def on_save_csv(self):
        if not self.last_data:
            return
        dec = int(self.decimales_var.get())
        df = pd.DataFrame(self.last_data, columns=["Iteración", "x_n", "g(x_n)", "Error_Abs", "Error_Rel", "x_Aitken"])
        for col in ["x_n", "g(x_n)", "Error_Abs", "Error_Rel", "x_Aitken"]:
            df[col] = df[col].apply(lambda v: f"{v:.{dec}f}" if v is not None else "")
        file = filedialog.asksaveasfilename(defaultextension=".csv")
        if file:
            df.to_csv(file, index=False, encoding="utf-8")

if __name__ == "__main__":
    app = App()
    app.mainloop()
