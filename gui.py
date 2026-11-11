import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sympy as sp
import math
import numpy as np
from matrix_solver import MatrixSolver
from golden_section import GoldenSectionSolver

# Available methods for non-linear equations
methods_data = {
    "Bisection": ["equation", "eps", "xl", "xu"],
    "False Position": ["equation", "eps", "xl", "xu"],
    "Fixed Point": ["equation", "eps", "x0"],
    "Newton-Raphson": ["equation", "eps", "x0"],
    "Secant": ["equation", "eps", "xiMinus1", "xi"]
}

def parse_function(expr):
    """Parse a mathematical expression safely using sympy."""
    x = sp.symbols('x')
    try:
        f_expr = sp.sympify(expr)
        return sp.lambdify(x, f_expr, modules=["math"])
    except Exception as e:
        raise ValueError(f"Error in function expression: {e}")

# Non-linear equation solving methods
def bisection_method(f_expr, xl, xu, eps, max_iter=100):
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, modules=["math"])
    iteration_data = []
    root_value = None
    iteration = 0
    while iteration < max_iter:
        xr = (xl + xu) / 2
        f_xr = f(xr)
        error = None if iteration == 0 else abs((xr - root_value) / xr) * 100 if root_value else 0
        iteration_data.append([iteration, f"{xl:.6f}", f"{xu:.6f}", f"{xr:.6f}", f"{error:.6f}" if error else "0"])
        if f(xr) == 0 or (error is not None and error < eps):
            root_value = xr
            break
        if f(xl) * f(xr) < 0:
            xu = xr
        else:
            xl = xr
        root_value = xr
        iteration += 1
    if iteration == max_iter:
        iteration_data.append(["Max iterations reached", "", "", "", ""])
    return iteration_data, root_value

def false_position_method(f_expr, xl, xu, eps, max_iter=100):
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, modules=["math"])
    iteration_data = []
    root_value = None
    iteration = 0
    while iteration < max_iter:
        f_xl = f(xl)
        f_xu = f(xu)
        xr = xu - (f_xu * (xl - xu)) / (f_xl - f_xu)
        f_xr = f(xr)
        error = None if iteration == 0 else abs((xr - root_value) / xr) * 100 if root_value else 0
        iteration_data.append([iteration, f"{xl:.6f}", f"{xu:.6f}", f"{xr:.6f}", f"{error:.6f}" if error else "0"])
        if f_xr == 0 or (error is not None and error < eps):
            root_value = xr
            break
        if f(xl) * f_xr < 0:
            xu = xr
        else:
            xl = xr
        root_value = xr
        iteration += 1
    if iteration == max_iter:
        iteration_data.append(["Max iterations reached", "", "", "", ""])
    return iteration_data, root_value

def fixed_point_method(f_expr, x0, eps, max_iter=100):
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, modules=["math"])
    g_expr = sp.sqrt(1.8 * x + 2.5)
    g = sp.lambdify(x, g_expr, modules=["math"])

    iteration = 0
    xi = x0
    error = None
    max_value = 1e10

    iteration_data = []
    root_value = None

    while iteration < max_iter:
        xi_plus1 = g(xi)
        
        if abs(xi_plus1) > max_value:
            iteration_data.append(["Overflow detected", "", "", "", ""])
            break

        f_xi = f(xi_plus1)

        if iteration > 0:
            error = abs((xi_plus1 - xi) / xi_plus1) * 100
        else:
            error = 0

        iteration_data.append([iteration, f"{xi:.6f}", f"{xi_plus1:.6f}", f"{f_xi:.6f}", f"{error:.6f}" if error else "0"])

        if error is not None and error <= eps and iteration != 0:
            root_value = xi_plus1
            break

        xi = xi_plus1
        iteration += 1

    if iteration == max_iter:
        iteration_data.append(["Max iterations reached", "", "", "", ""])
    
    if root_value is None:
        root_value = xi

    return iteration_data, root_value

def newton_raphson_method(f_expr, x0, eps, max_iter=100):
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, modules=["math"])
    df_expr = sp.diff(f_expr, x)
    df = sp.lambdify(x, df_expr, modules=["math"])
    iteration_data = []
    root_value = x0
    iteration = 0
    while iteration < max_iter:
        f_x = f(root_value)
        df_x = df(root_value)
        if df_x == 0:
            iteration_data.append(["Division by zero in derivative", "", "", "", ""])
            break
        xi = root_value - f_x / df_x
        error = abs((xi - root_value) / xi) * 100 if xi != 0 else 0
        iteration_data.append([iteration, f"{root_value:.6f}", f"{xi:.6f}", f"{f_x:.6f}", f"{error:.6f}" if error else "0"])
        if error < eps:
            root_value = xi
            break
        root_value = xi
        iteration += 1
    if iteration == max_iter:
        iteration_data.append(["Max iterations reached", "", "", "", ""])
    return iteration_data, root_value

def secant_method(f_expr, x0, x1, eps, max_iter=100):
    x = sp.symbols('x')
    f = sp.lambdify(x, f_expr, modules=["math"])
    iteration_data = []
    root_value = x1
    iteration = 0
    while iteration < max_iter:
        f_x0 = f(x0)
        f_x1 = f(x1)
        if f_x1 - f_x0 == 0:
            iteration_data.append(["Division by zero", "", "", "", ""])
            break
        xi = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        error = abs((xi - x1) / xi) * 100 if xi != 0 else 0
        iteration_data.append([iteration, f"{x0:.6f}", f"{x1:.6f}", f"{f_x1:.6f}", f"{error:.6f}" if error else "0"])
        if error < eps:
            root_value = xi
            break
        x0 = x1
        x1 = xi
        iteration += 1
    if iteration == max_iter:
        iteration_data.append(["Max iterations reached", "", "", "", ""])
    return iteration_data, root_value


def apply_professional_style(root):
    """Apply a modern dark AI-style theme without changing solver logic."""
    BG = "#0B1020"
    PANEL = "#151D32"
    PANEL_ALT = "#1B2742"
    BLUE = "#397BFF"
    BLUE_HOVER = "#5590FF"
    TEXT = "#EAF1FF"
    MUTED = "#A9B8D0"
    INPUT = "#10192B"
    BORDER = "#2B3A59"

    root.configure(bg=BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure(".", font=("Segoe UI", 10))
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 11))
    style.configure("TRadiobutton", background=PANEL, foreground=TEXT,
                    font=("Segoe UI", 11), padding=10)
    style.map("TRadiobutton", background=[("active", PANEL_ALT)],
              foreground=[("selected", BLUE), ("!selected", TEXT)])
    style.configure("TButton", background=BLUE, foreground="#FFFFFF",
                    font=("Segoe UI", 10, "bold"), padding=(18, 10),
                    borderwidth=0, relief="flat")
    style.map("TButton", background=[("active", BLUE_HOVER), ("pressed", "#2459C7")],
              foreground=[("disabled", "#8290A8"), ("!disabled", "#FFFFFF")])
    style.configure("TEntry", fieldbackground=INPUT, foreground=TEXT,
                    insertcolor=TEXT, padding=9, bordercolor=BORDER,
                    lightcolor=BORDER, darkcolor=BORDER)
    style.configure("TCombobox", fieldbackground=INPUT, background=PANEL_ALT,
                    foreground=TEXT, arrowcolor=TEXT, padding=9, bordercolor=BORDER)
    style.map("TCombobox", fieldbackground=[("readonly", INPUT), ("focus", INPUT)],
              foreground=[("readonly", TEXT)], selectbackground=[("readonly", INPUT)],
              selectforeground=[("readonly", TEXT)])
    style.configure("Custom.TLabelframe", background=PANEL, foreground=TEXT,
                    borderwidth=1, relief="solid")
    style.configure("Custom.TLabelframe.Label", background=PANEL,
                    foreground=BLUE_HOVER, font=("Segoe UI", 11, "bold"))
    style.configure("TLabelframe", background=PANEL, foreground=TEXT,
                    borderwidth=1, relief="solid")
    style.configure("TLabelframe.Label", background=PANEL,
                    foreground=BLUE_HOVER, font=("Segoe UI", 11, "bold"))
    style.configure("Treeview", background=INPUT, foreground=TEXT,
                    fieldbackground=INPUT, rowheight=32, font=("Segoe UI", 10),
                    borderwidth=0)
    style.configure("Treeview.Heading", background=PANEL_ALT, foreground=TEXT,
                    font=("Segoe UI", 10, "bold"), padding=10, relief="flat")
    style.map("Treeview", background=[("selected", "#244B8F")],
              foreground=[("selected", "#FFFFFF")])
    style.configure("TNotebook", background=BG, borderwidth=0)
    style.configure("TNotebook.Tab", background=PANEL_ALT, foreground=MUTED,
                    padding=(18, 11), font=("Segoe UI", 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", BLUE)],
              foreground=[("selected", "#FFFFFF")])

    title = root.title()
    if title == "Equation Solver":
        root.geometry("900x620")
        root.minsize(760, 540)
    elif "Non-Linear" in title or "Golden Section" in title:
        root.geometry("1100x760")
        root.minsize(900, 650)
    elif "Linear" in title:
        root.geometry("1150x800")
        root.minsize(950, 680)

    def style_children(parent):
        for widget in parent.winfo_children():
            try:
                cls = widget.winfo_class()
                if isinstance(widget, tk.Label):
                    widget.configure(bg=PANEL, fg=TEXT, font=("Segoe UI", 11, "bold"),
                                     padx=8, pady=5)
                elif isinstance(widget, tk.Text):
                    widget.configure(bg=INPUT, fg=TEXT, insertbackground=TEXT,
                                     selectbackground="#244B8F", relief="flat",
                                     borderwidth=1, font=("Consolas", 10))
                elif isinstance(widget, ttk.Label):
                    widget.configure(background=PANEL, foreground=TEXT)
                elif isinstance(widget, ttk.LabelFrame):
                    widget.configure(style="Custom.TLabelframe")
                elif isinstance(widget, ttk.Frame):
                    widget.configure(style="TFrame")
                elif isinstance(widget, ttk.Treeview):
                    widget.tag_configure("odd", background=INPUT, foreground=TEXT)
                    widget.tag_configure("even", background=PANEL, foreground=TEXT)
                elif cls == "Frame":
                    widget.configure(bg=BG)
            except (tk.TclError, TypeError):
                pass
            style_children(widget)

    style_children(root)

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Equation Solver")
        self.root.geometry("600x400")
        
        # Set background color
        self.root.configure(bg='#F4F7FC')  # Light blue background
        
        # Configure ttk Style
        style = ttk.Style()
        style.configure("TLabel", font=("- Segoe UI", 14, "bold"), background='#F4F7FC', foreground='#172B4D')
        style.configure("TRadiobutton", font=("- Segoe UI", 12), background='#F4F7FC', foreground='#334155')
        style.configure("TButton", font=("- Segoe UI", 12, "bold"))
        style.map("TButton", background=[('active', '#1D4ED8'), ('!active', '#2563EB')],
                  foreground=[('active', '#172B4D'), ('!active', '#172B4D')])
        style.configure("TFrame", background='#F4F7FC')
        
        self.choice_var = tk.StringVar(value="Non-Linear")
        
        self.create_widgets()
        self.setup_layout()
        self.root.after_idle(lambda: apply_professional_style(self.root))
    
    def create_widgets(self):
        # Frame for selection with border
        self.selection_frame = ttk.LabelFrame(self.root, text="Equation Type Selection", padding=20, style="Custom.TLabelframe")
        self.selection_frame.pack(pady=20, padx=20, fill='both', expand=True)
        
        # Custom style for LabelFrame
        style = ttk.Style()
        style.configure("Custom.TLabelframe", background='#FFFFFF', foreground='#172B4D')
        style.configure("Custom.TLabelframe.Label", font=("- Segoe UI", 14, "bold"), foreground='#172B4D')

        # Title
        ttk.Label(self.selection_frame, text="Select Equation Type:", font=("- Segoe UI", 16, "bold"), background='#FFFFFF', foreground='#172B4D').pack(pady=15, anchor='center')
        
        # Radio buttons with spacing
        ttk.Radiobutton(self.selection_frame, text="Non-Linear Equation", value="Non-Linear", variable=self.choice_var).pack(pady=10, anchor='center')
        ttk.Radiobutton(self.selection_frame, text="Matrix Linear System", value="Matrix", variable=self.choice_var).pack(pady=10, anchor='center')
        ttk.Radiobutton(self.selection_frame, text="Golden Section Search", value="Golden", variable=self.choice_var).pack(pady=10, anchor='center')
        
        # Proceed button
        self.proceed_button = ttk.Button(self.selection_frame, text="Proceed", command=self.launch_solver)
        self.proceed_button.pack(pady=20, anchor='center')
    
    def setup_layout(self):
        pass
    
    def launch_solver(self):
        choice = self.choice_var.get()
        self.root.destroy()
        
        new_root = tk.Tk()
        if choice == "Non-Linear":
            app = NumericalApp(new_root)
        elif choice == "Matrix":
            app = LinearSolverApp(new_root)
        else:
            app = GoldenSectionApp(new_root)
        new_root.mainloop()

class NumericalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Non-Linear Equation Solver")
        self.root.geometry("600x500")

        # Set background color
        self.root.configure(bg='#F4F7FC')

        # Configure ttk Style
        style = ttk.Style()
        style.configure("TLabel", font=("- Segoe UI", 12, "bold"), background='#F4F7FC', foreground='#172B4D')
        style.configure("TCombobox", font=("- Segoe UI", 12))
        style.map("TCombobox", fieldbackground=[('readonly', '#FFFFFF')])
        style.configure("TButton", font=("- Segoe UI", 12, "bold"))
        style.map("TButton", background=[('active', '#1D4ED8'), ('!active', '#2563EB')],
                  foreground=[('active', '#172B4D'), ('!active', '#172B4D')])
        style.configure("TFrame", background='#F4F7FC')
        style.configure("Custom.TLabelframe", background='#FFFFFF', foreground='#172B4D')
        style.configure("Custom.TLabelframe.Label", font=("- Segoe UI", 12, "bold"), foreground='#172B4D')
        style.configure("Treeview", font=("- Segoe UI", 11), background='#FFFFFF', foreground='#172B4D', fieldbackground='#FFFFFF')
        style.configure("Treeview.Heading", font=("- Segoe UI", 12, "bold"), background='#172B4D', foreground='#172B4D')  # Changed to darker color

        self.fields = {}
        self.method_var = tk.StringVar(value="")

        self.create_widgets()
        self.setup_layout()
        self.root.after_idle(lambda: apply_professional_style(self.root))

    def create_widgets(self):
        # Back to Launcher button
        self.back_button = ttk.Button(self.root, text="Back to Methods", command=self.back_to_launcher)
        self.back_button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

        # Method selection
        ttk.Label(self.root, text="Choose Method:", font=("- Segoe UI", 14, "bold")).pack(pady=10, anchor='center')
        self.method_menu = ttk.Combobox(self.root, textvariable=self.method_var, values=list(methods_data.keys()), state="readonly", width=30)
        self.method_menu.pack(pady=5, anchor='center')
        self.method_menu.bind("<<ComboboxSelected>>", self.update_fields)

        # Input fields frame
        self.fields_frame = ttk.LabelFrame(self.root, text="Input Parameters", padding=15, style="Custom.TLabelframe")
        self.fields_frame.pack(pady=10, padx=20, fill='x')

        # Buttons frame
        self.buttons_frame = ttk.Frame(self.root, style="TFrame")
        self.buttons_frame.pack(pady=10)

        self.run_button = ttk.Button(self.buttons_frame, text="Run", command=self.run_method)
        self.run_button.pack(side=tk.LEFT, padx=15)

        self.reset_button = ttk.Button(self.buttons_frame, text="Reset", command=self.reset_all)
        self.reset_button.pack(side=tk.LEFT, padx=15)

        # Table frame
        self.table_frame = ttk.Frame(self.root, style="TFrame")
        self.table_frame.pack(pady=10, padx=20, fill='both', expand=True)

        # Result label
        self.result_box = tk.Label(self.root, text="Required Root: ", font=("- Segoe UI", 12, "bold"), bg='#F4F7FC', fg='#172B4D')
        self.result_box.pack(pady=10, anchor='center')

    def setup_layout(self):
        pass

    def update_fields(self, event=None):
        for widget in self.fields_frame.winfo_children():
            widget.destroy()
        self.fields.clear()

        selected_method = self.method_var.get()
        if not selected_method:
            return

        required_fields = methods_data.get(selected_method, [])
        for field in required_fields:
            label_text = "Equation f(x):" if field == "equation" else f"{field.capitalize()}:"
            label = ttk.Label(self.fields_frame, text=label_text, font=("- Segoe UI", 12))
            label.pack(pady=5, anchor='center')
            entry = ttk.Entry(self.fields_frame, width=30, font=("- Segoe UI", 12))
            entry.pack(pady=5, anchor='center')
            self.fields[field] = entry

    def run_method(self):
        method = self.method_var.get()
        if not method:
            messagebox.showerror("Error", "Please select a method.")
            return

        try:
            inputs = {key: entry.get() for key, entry in self.fields.items()}
            f_expr = sp.sympify(inputs["equation"])
            eps = float(inputs["eps"])

            if method == "Bisection":
                xl = float(inputs["xl"])
                xu = float(inputs["xu"])
                iteration_data, root_value = bisection_method(f_expr, xl, xu, eps)
            elif method == "False Position":
                xl = float(inputs["xl"])
                xu = float(inputs["xu"])
                iteration_data, root_value = false_position_method(f_expr, xl, xu, eps)
            elif method == "Fixed Point":
                x0 = float(inputs["x0"])
                iteration_data, root_value = fixed_point_method(f_expr, x0, eps)
            elif method == "Newton-Raphson":
                x0 = float(inputs["x0"])
                iteration_data, root_value = newton_raphson_method(f_expr, x0, eps)
            elif method == "Secant":
                xi_minus1 = float(inputs["xiMinus1"])
                xi = float(inputs["xi"])
                iteration_data, root_value = secant_method(f_expr, xi_minus1, xi, eps)
            else:
                messagebox.showerror("Error", "Invalid method selected.")
                return

            for widget in self.table_frame.winfo_children():
                widget.destroy()

            if iteration_data:
                headers = ["i", "Xi", "g(Xi)/Xr", "f(Xi)", "Error %"]
                tree = ttk.Treeview(self.table_frame, columns=headers, show="headings")
                tree.pack(fill='both', expand=True)

                window_width = self.root.winfo_width()
                column_width = max(100, window_width // len(headers) - 20)
                for col in headers:
                    tree.heading(col, text=col)
                    tree.column(col, anchor="center", width=column_width)

                for row in iteration_data:
                    tree.insert("", "end", values=row)

            self.result_box.config(text=f"Required Root: {root_value:.6f}" if root_value else "Required Root: Not found")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_all(self):
        for field in self.fields.values():
            field.delete(0, tk.END)
        self.method_var.set("")
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self.result_box.config(text="Required Root: ")
        self.update_fields()

    def back_to_launcher(self):
        self.root.destroy()
        new_root = tk.Tk()
        app = LauncherApp(new_root)
        new_root.mainloop()

class LinearSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Equation Solver")
        self.root.geometry("800x600")
        
        # Set background color
        self.root.configure(bg='#F4F7FC')

        # Configure ttk Style
        style = ttk.Style()
        style.configure("TLabel", font=("- Segoe UI", 12, "bold"), background='#F4F7FC', foreground='#172B4D')
        style.configure("TCombobox", font=("- Segoe UI", 12))
        style.map("TCombobox", fieldbackground=[('readonly', '#FFFFFF')])
        style.configure("TButton", font=("- Segoe UI", 12, "bold"))
        style.map("TButton", background=[('active', '#1D4ED8'), ('!active', '#2563EB')],
                  foreground=[('active', '#172B4D'), ('!active', '#172B4D')])
        style.configure("TFrame", background='#F4F7FC')
        style.configure("Custom.TLabelframe", background='#FFFFFF', foreground='#172B4D')
        style.configure("Custom.TLabelframe.Label", font=("- Segoe UI", 12, "bold"), foreground='#172B4D')
        style.configure("TNotebook", background='#FFFFFF')
        style.configure("TNotebook.Tab", font=("- Segoe UI", 12, "bold"), background='#2563EB', foreground='#FFFFFF')
        
        self.create_widgets()
        self.setup_layout()
        self.root.after_idle(lambda: apply_professional_style(self.root))
    
    def create_widgets(self):
        # Back to Launcher button
        self.back_button = ttk.Button(self.root, text="Back to Methods", command=self.back_to_launcher)
        self.back_button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

        # Matrix input frame
        self.matrix_frame = ttk.LabelFrame(self.root, text="Matrix Input", padding=15, style="Custom.TLabelframe")
        self.size_var = tk.IntVar(value=3)
        self.size_selector = ttk.Combobox(self.matrix_frame, textvariable=self.size_var, 
                                         values=[2, 3, 4, 5], state='readonly', width=10)
        self.size_selector.bind('<<ComboboxSelected>>', self.update_matrix_input)
        
        self.matrix_entries = []
        self.create_matrix_input()
        
        # Method selection
        self.method_frame = ttk.LabelFrame(self.root, text="Solving Method", padding=15, style="Custom.TLabelframe")
        self.method_var = tk.StringVar()
        self.methods = [
            "Gauss Elimination",
            "Gauss Elimination with Partial Pivoting",
            "Gauss-Jordan",
            "Gauss-Jordan with Partial Pivoting",
            "LU Decomposition",
            "LU Decomposition with Partial Pivoting",
            "Cramer's Rule"
        ]
        self.method_selector = ttk.Combobox(self.method_frame, textvariable=self.method_var, 
                                          values=self.methods, state='readonly', width=30)
        self.method_selector.current(0)
        
        # Buttons
        self.solve_btn = ttk.Button(self.root, text="Solve", command=self.solve)
        self.clear_btn = ttk.Button(self.root, text="Clear", command=self.clear)
        
        # Results display
        self.result_frame = ttk.LabelFrame(self.root, text="Results", padding=15, style="Custom.TLabelframe")
        self.solution_text = scrolledtext.ScrolledText(self.result_frame, height=10, wrap=tk.WORD, font=("- Segoe UI", 12))
        self.steps_text = scrolledtext.ScrolledText(self.result_frame, height=15, wrap=tk.WORD, font=("- Segoe UI", 12))
        
        # Tabs for results and steps
        self.notebook = ttk.Notebook(self.result_frame)
        self.solution_tab = ttk.Frame(self.notebook)
        self.steps_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.solution_tab, text="Solution")
        self.notebook.add(self.steps_tab, text="Solving Steps")
        
        self.solution_text_tab = scrolledtext.ScrolledText(self.solution_tab, wrap=tk.WORD, font=("- Segoe UI", 12))
        self.steps_text_tab = scrolledtext.ScrolledText(self.steps_tab, wrap=tk.WORD, font=("- Segoe UI", 12))
        
        self.solution_text_tab.pack(expand=True, fill='both')
        self.steps_text_tab.pack(expand=True, fill='both')
    
    def setup_layout(self):
        self.matrix_frame.pack(pady=15, padx=20, fill='x')
        ttk.Label(self.matrix_frame, text="Matrix Size:", font=("- Segoe UI", 12)).grid(row=0, column=0, padx=10, pady=10)
        self.size_selector.grid(row=0, column=1, padx=10, pady=10)
        
        self.method_frame.pack(pady=15, padx=20, fill='x')
        ttk.Label(self.method_frame, text="Select Solving Method:", font=("- Segoe UI", 12)).grid(row=0, column=0, padx=10, pady=10)
        self.method_selector.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Frame(self.root, height=10, style="TFrame").pack()
        
        self.solve_btn.pack(side='left', padx=30, pady=10)
        self.clear_btn.pack(side='right', padx=30, pady=10)
        
        self.result_frame.pack(pady=15, padx=20, expand=True, fill='both')
        self.notebook.pack(expand=True, fill='both')
    
    def create_matrix_input(self):
        for widget in self.matrix_frame.winfo_children():
            if widget not in [self.size_selector]:
                widget.destroy()
        
        n = self.size_var.get()
        self.matrix_entries = []
        
        for j in range(n+1):
            header = f"x{j+1}" if j < n else "b"
            ttk.Label(self.matrix_frame, text=header, font=("- Segoe UI", 12)).grid(row=1, column=j+1, padx=5, pady=5)
        
        for i in range(n):
            row_entries = []
            ttk.Label(self.matrix_frame, text=f"Equation {i+1}", font=("- Segoe UI", 12)).grid(row=i+2, column=0, padx=5, pady=5)
            
            for j in range(n+1):
                entry = ttk.Entry(self.matrix_frame, width=8, font=("- Segoe UI", 12))
                entry.grid(row=i+2, column=j+1, padx=5, pady=5)
                row_entries.append(entry)
            
            self.matrix_entries.append(row_entries)
    
    def update_matrix_input(self, event=None):
        self.create_matrix_input()
    
    def get_matrix(self):
        n = self.size_var.get()
        matrix = []
        
        for i in range(n):
            row = []
            for j in range(n+1):
                try:
                    val = float(self.matrix_entries[i][j].get())
                    row.append(val)
                except ValueError:
                    messagebox.showerror("Input Error", "Please enter valid numbers only")
                    return None
            matrix.append(row)
        
        return matrix
    
    def solve(self):
        matrix = self.get_matrix()
        if matrix is None:
            return
        
        method = self.method_var.get()
        solver = MatrixSolver(matrix)
        
        try:
            if method == "Gauss Elimination":
                solution, steps = solver.gauss_elimination()
            elif method == "Gauss Elimination with Partial Pivoting":
                solution, steps = solver.gauss_pivoting()
            elif method == "Gauss-Jordan":
                solution, steps = solver.gauss_jordan()
            elif method == "Gauss-Jordan with Partial Pivoting":
                solution, steps = solver.gauss_jordan_pivoting()
            elif method == "LU Decomposition":
                solution, steps = solver.lu_decomposition()
            elif method == "LU Decomposition with Partial Pivoting":
                solution, steps = solver.lu_pivoting()
            elif method == "Cramer's Rule":
                solution, steps = solver.cramers_rule()
            else:
                messagebox.showerror("Error", "Please select a solving method")
                return
        except Exception as e:
            messagebox.showerror("Solving Error", str(e))
            return
        
        self.solution_text_tab.delete(1.0, tk.END)
        if solution is None:
            self.solution_text_tab.insert(tk.END, "No unique solution exists for the system")
        else:
            for i, val in enumerate(solution):
                self.solution_text_tab.insert(tk.END, f"x{i+1} = {val:.6f}\n")
        
        self.steps_text_tab.delete(1.0, tk.END)
        for step in steps:
            self.steps_text_tab.insert(tk.END, step + "\n" + "-"*50 + "\n")

    def clear(self):
        self.solution_text_tab.delete(1.0, tk.END)
        self.steps_text_tab.delete(1.0, tk.END)
        for row in self.matrix_entries:
            for entry in row:
                entry.delete(0, tk.END)

    def back_to_launcher(self):
        self.root.destroy()
        new_root = tk.Tk()
        app = LauncherApp(new_root)
        new_root.mainloop()

class GoldenSectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Golden Section Search")
        self.root.geometry("800x600")
        
        # Set background color
        self.root.configure(bg='#F4F7FC')

        # Configure ttk Style
        style = ttk.Style()
        style.configure("TLabel", font=("- Segoe UI", 12, "bold"), background='#F4F7FC', foreground='#172B4D')
        style.configure("TCombobox", font=("- Segoe UI", 12))
        style.map("TCombobox", fieldbackground=[('readonly', '#FFFFFF')])
        style.configure("TButton", font=("- Segoe UI", 12, "bold"))
        style.map("TButton", background=[('active', '#1D4ED8'), ('!active', '#2563EB')],
                  foreground=[('active', '#172B4D'), ('!active', '#172B4D')])
        style.configure("TFrame", background='#F4F7FC')
        style.configure("Custom.TLabelframe", background='#FFFFFF', foreground='#172B4D')
        style.configure("Custom.TLabelframe.Label", font=("- Segoe UI", 12, "bold"), foreground='#172B4D')
        style.configure("Treeview", font=("- Segoe UI", 11), background='#FFFFFF', foreground='#172B4D', fieldbackground='#FFFFFF')
        style.configure("Treeview.Heading", font=("- Segoe UI", 12, "bold"), background='#172B4D', foreground='#172B4D')  # Changed to darker color

        self.fields = {}
        self.mode_var = tk.StringVar(value="Maximum")
        self.create_widgets()
        self.setup_layout()
        self.root.after_idle(lambda: apply_professional_style(self.root))

    def create_widgets(self):
        # Back to Launcher button
        self.back_button = ttk.Button(self.root, text="Back to Methods", command=self.back_to_launcher)
        self.back_button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

        self.fields_frame = ttk.LabelFrame(self.root, text="Input Parameters", padding=15, style="Custom.TLabelframe")
        self.fields_frame.pack(pady=10, padx=20, fill='x')

        # Mode selection
        ttk.Label(self.fields_frame, text="Select Goal:", font=("- Segoe UI", 12)).pack(pady=5, anchor='center')
        self.mode_menu = ttk.Combobox(self.fields_frame, textvariable=self.mode_var, values=["Maximum", "Minimum"], state="readonly", width=30)
        self.mode_menu.pack(pady=5, anchor='center')

        # Input fields
        fields = ["function", "xl", "xu"]
        for field in fields:
            label_text = "Function f(x):" if field == "function" else f"{field.capitalize()}:"
            label = ttk.Label(self.fields_frame, text=label_text, font=("- Segoe UI", 12))
            label.pack(pady=5, anchor='center')
            entry = ttk.Entry(self.fields_frame, width=30, font=("- Segoe UI", 12))
            entry.pack(pady=5, anchor='center')
            self.fields[field] = entry

        # Buttons frame
        self.buttons_frame = ttk.Frame(self.root, style="TFrame")
        self.buttons_frame.pack(pady=10)
        self.run_button = ttk.Button(self.buttons_frame, text="Run", command=self.run_method)
        self.run_button.pack(side=tk.LEFT, padx=15)
        self.reset_button = ttk.Button(self.buttons_frame, text="Reset", command=self.reset_all)
        self.reset_button.pack(side=tk.LEFT, padx=15)

        # Table frame
        self.table_frame = ttk.Frame(self.root, style="TFrame")
        self.table_frame.pack(pady=10, padx=20, fill='both', expand=True)

        # Result label
        self.result_box = tk.Label(self.root, text="Result: ", font=("- Segoe UI", 12, "bold"), bg='#F4F7FC', fg='#172B4D')
        self.result_box.pack(pady=10, anchor='center')

    def setup_layout(self):
        pass

    def run_method(self):
        try:
            inputs = {key: entry.get() for key, entry in self.fields.items()}
            expr = inputs["function"]
            xl = float(inputs["xl"])
            xu = float(inputs["xu"])
            if xl >= xu:
                raise ValueError("XL must be less than XU.")
            func = parse_function(expr)
            solver = GoldenSectionSolver(func, xl, xu)

            maximize = self.mode_var.get() == "Maximum"
            iteration_data, value = solver.golden_section_search(maximize=maximize)

            for widget in self.table_frame.winfo_children():
                widget.destroy()

            if iteration_data:
                headers = ["i", "x_l", "f(x_l)", "x_2", "f(x_2)", "x_1", "f(x_1)", "x_u", "f(x_u)", "d"]
                tree = ttk.Treeview(self.table_frame, columns=headers, show="headings")
                tree.pack(fill='both', expand=True)
                window_width = self.root.winfo_width()
                column_width = max(70, window_width // len(headers) - 20)
                for col in headers:
                    tree.heading(col, text=col)
                    tree.column(col, anchor="center", width=column_width)
                for row in iteration_data:
                    tree.insert("", "end", values=row)

            steps = solver.steps
            if steps:
                result_message = steps[-1]
                expected_label = "Maximum" if maximize else "Minimum"
                if not result_message.startswith(expected_label):
                    value_part = result_message.split(" at x = ")[1]
                    result_message = f"{expected_label} at x = {value_part}"
            else:
                result_message = "Result: Not found"

            self.result_box.config(text=f"Result: {result_message}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def reset_all(self):
        for field in self.fields.values():
            field.delete(0, tk.END)
        self.mode_var.set("Maximum")
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self.result_box.config(text="Result: ")

    def back_to_launcher(self):
        self.root.destroy()
        new_root = tk.Tk()
        app = LauncherApp(new_root)
        new_root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()