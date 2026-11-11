
import ast
import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st

# ==========================================
# NumeraX - Numerical Analysis Web Application
# ==========================================

st.set_page_config(
    page_title="NumeraX | Numerical Computing",
    page_icon="📐",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color: #17365D;
}
</style>
""", unsafe_allow_html=True)

x = sp.Symbol("x")

st.title("📐 NumeraX")
st.subheader("Numerical Analysis & Computing System")
st.write(
    "An interactive platform for solving mathematical "
    "problems using numerical methods."
)

st.divider()

# ==========================================
# Safe mathematical expression parser
# ==========================================

FUNCTIONS = {
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "abs": sp.Abs
}

def safe_expression(node):
    if isinstance(node, ast.Expression):
        return safe_expression(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return sp.Number(node.value)
        raise ValueError("Unsupported constant.")

    if isinstance(node, ast.Name):
        if node.id == "x":
            return x
        if node.id == "pi":
            return sp.pi
        if node.id == "e":
            return sp.E
        raise ValueError(f"Unknown variable: {node.id}")

    if isinstance(node, ast.BinOp):
        left = safe_expression(node.left)
        right = safe_expression(node.right)

        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left ** right
        }

        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported mathematical operator.")
        return operation()

    if isinstance(node, ast.UnaryOp):
        value = safe_expression(node.operand)

        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function.")

        function = FUNCTIONS.get(node.func.id)
        if function is None or node.keywords:
            raise ValueError("Unsupported mathematical function.")

        arguments = [safe_expression(arg) for arg in node.args]
        return function(*arguments)

    raise ValueError("Invalid mathematical expression.")


def build_function(expression):
    tree = ast.parse(expression, mode="eval")
    symbolic_expression = safe_expression(tree)
    return sp.lambdify(x, symbolic_expression, modules=["numpy"])


def evaluate(function, value):
    result = float(function(value))

    if not np.isfinite(result):
        raise ValueError("Function produced an invalid numerical value.")

    return result


# ==========================================
# Numerical Methods
# ==========================================

def bisection(f, xl, xu, tolerance, max_iter):
    fl = evaluate(f, xl)
    fu = evaluate(f, xu)

    if fl * fu >= 0:
        raise ValueError("The interval must have opposite signs.")

    rows = []
    previous = None

    for i in range(1, max_iter + 1):
        xr = (xl + xu) / 2
        fr = evaluate(f, xr)

        error = abs(xr - previous) if previous is not None else None

        rows.append({
            "Iteration": i,
            "XL": xl,
            "XU": xu,
            "XR": xr,
            "f(XR)": fr,
            "Error": error
        })

        if abs(fr) < tolerance or (error is not None and error < tolerance):
            return xr, rows

        if fl * fr < 0:
            xu = xr
            fu = fr
        else:
            xl = xr
            fl = fr

        previous = xr

    return xr, rows


def false_position(f, xl, xu, tolerance, max_iter):
    fl = evaluate(f, xl)
    fu = evaluate(f, xu)

    if fl * fu >= 0:
        raise ValueError("The interval must have opposite signs.")

    rows = []
    previous = None

    for i in range(1, max_iter + 1):
        denominator = fl - fu

        if abs(denominator) < 1e-14:
            raise ValueError("Division by zero in False Position.")

        xr = xu - fu * (xl - xu) / denominator
        fr = evaluate(f, xr)

        error = abs(xr - previous) if previous is not None else None

        rows.append({
            "Iteration": i,
            "XL": xl,
            "XU": xu,
            "XR": xr,
            "f(XR)": fr,
            "Error": error
        })

        if abs(fr) < tolerance or (error is not None and error < tolerance):
            return xr, rows

        if fl * fr < 0:
            xu = xr
            fu = fr
        else:
            xl = xr
            fl = fr

        previous = xr

    return xr, rows


def fixed_point(tolerance, max_iter, initial):
    rows = []

    current = initial

    for i in range(1, max_iter + 1):
        if 1.8 * current + 2.5 < 0:
            raise ValueError("Square root is undefined for this input.")

        next_value = np.sqrt(1.8 * current + 2.5)
        error = abs(next_value - current)

        rows.append({
            "Iteration": i,
            "Old Value": current,
            "New Value": next_value,
            "Error": error
        })

        current = next_value

        if error < tolerance:
            return current, rows

    return current, rows


def newton_raphson(f, expression, initial, tolerance, max_iter):
    derivative = sp.diff(expression, x)
    df = sp.lambdify(x, derivative, modules=["numpy"])

    current = initial
    rows = []

    for i in range(1, max_iter + 1):
        fx = evaluate(f, current)
        dfx = evaluate(df, current)

        if abs(dfx) < 1e-14:
            raise ValueError("Derivative is too close to zero.")

        next_value = current - fx / dfx
        error = abs(next_value - current)

        rows.append({
            "Iteration": i,
            "X": current,
            "f(X)": fx,
            "f'(X)": dfx,
            "Next X": next_value,
            "Error": error
        })

        current = next_value

        if error < tolerance or abs(evaluate(f, current)) < tolerance:
            return current, rows

    return current, rows


def secant(f, x0, x1, tolerance, max_iter):
    rows = []

    for i in range(1, max_iter + 1):
        f0 = evaluate(f, x0)
        f1 = evaluate(f, x1)

        denominator = f1 - f0

        if abs(denominator) < 1e-14:
            raise ValueError("Division by zero in Secant Method.")

        next_value = x1 - f1 * (x1 - x0) / denominator
        error = abs(next_value - x1)

        rows.append({
            "Iteration": i,
            "X0": x0,
            "X1": x1,
            "Next X": next_value,
            "Error": error
        })

        x0, x1 = x1, next_value

        if error < tolerance or abs(evaluate(f, x1)) < tolerance:
            return x1, rows

    return x1, rows


# ==========================================
# Application Interface
# ==========================================

st.sidebar.title("Navigation")

category = st.sidebar.selectbox(
    "Select Category",
    ["Nonlinear Equations"]
)

method = st.sidebar.selectbox(
    "Select Numerical Method",
    [
        "Bisection Method",
        "False Position Method",
        "Fixed Point Iteration",
        "Newton-Raphson Method",
        "Secant Method"
    ]
)

st.header(method)

st.info(
    "Enter the mathematical equation and numerical parameters "
    "to calculate the solution."
)

with st.form("solver_form"):

    equation = st.text_input(
        "Equation f(x)",
        value="x**2 - 4",
        help="Example: x**2 - 4, sin(x), exp(x) - 2"
    )

    col1, col2 = st.columns(2)

    with col1:
        xl = st.number_input("Lower Bound (XL)", value=1.0)
        x0 = st.number_input("Initial Value (X0)", value=1.0)
        tolerance = st.number_input(
            "Tolerance",
            min_value=0.0000001,
            value=0.001,
            format="%.7f"
        )

    with col2:
        xu = st.number_input("Upper Bound (XU)", value=3.0)
        x1 = st.number_input("Second Initial Value (X1)", value=3.0)
        max_iter = st.number_input(
            "Maximum Iterations",
            min_value=1,
            max_value=500,
            value=50
        )

    submitted = st.form_submit_button(
        "🚀 Solve Equation",
        use_container_width=True
    )


if submitted:
    try:
        expression = safe_expression(
            ast.parse(equation, mode="eval")
        )
        f = build_function(equation)

        if method == "Bisection Method":
            result, rows = bisection(
                f, xl, xu, tolerance, int(max_iter)
            )

        elif method == "False Position Method":
            result, rows = false_position(
                f, xl, xu, tolerance, int(max_iter)
            )

        elif method == "Fixed Point Iteration":
            result, rows = fixed_point(
                tolerance, int(max_iter), x0
            )

        elif method == "Newton-Raphson Method":
            result, rows = newton_raphson(
                f, expression, x0, tolerance, int(max_iter)
            )

        else:
            result, rows = secant(
                f, x0, x1, tolerance, int(max_iter)
            )

        st.success("Calculation completed!")

        st.subheader("Numerical Result")

        col1, col2 = st.columns(2)

        col1.metric("Approximate Root", f"{result:.10f}")
        col2.metric("Iterations", len(rows))

        st.subheader("Iteration Table")
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True
        )

        if method != "Fixed Point Iteration":
            st.subheader("Function")
            st.latex(sp.latex(expression))

            st.write(
                "Final function value:",
                f"{evaluate(f, result):.10f}"
            )
        else:
            st.caption(
                "Fixed Point uses the predefined function "
                "g(x) = sqrt(1.8x + 2.5)."
            )

    except Exception as error:
        st.error(f"Calculation Error: {error}")


st.divider()

st.caption(
    "NumeraX | Numerical Analysis & Computing System"
)
st.caption(
    "Educational project — Python, Streamlit, NumPy and SymPy"
)
