# Equation Solver

A desktop application for solving mathematical equations using a graphical user interface (GUI) built with Python and Tkinter.

The application provides **13 numerical methods** across three main categories: nonlinear equations, linear systems, and optimization.

## Features

- User-friendly graphical interface.
- Multiple numerical methods in one application.
- Iteration tables displaying calculation results.
- Detailed solving steps for matrix methods.
- Input validation and error handling.
- Symbolic mathematical expression processing using SymPy.

## Numerical Methods

### 1. Nonlinear Equations

Implemented in `gui.py`.

1. Bisection Method
2. False Position Method
3. Fixed Point Iteration
4. Newton-Raphson Method
5. Secant Method

### 2. Linear Systems

Implemented in `matrix_solver.py`.

1. Gauss Elimination
2. Gauss Elimination with Partial Pivoting
3. Gauss-Jordan Elimination
4. Gauss-Jordan with Partial Pivoting
5. LU Decomposition
6. LU Decomposition with Partial Pivoting
7. Cramer's Rule

### 3. Optimization

Implemented in `golden_section.py`.

1. Golden Section Search

## Project Structure

```text
Numerical projct/
│
├── gui.py
├── matrix_solver.py
├── golden_section.py
├── requirements.txt
└── README.md
```

Note: Archived and backup files are kept locally and are not included in the GitHub repository.

## Technologies Used

- Python
- Tkinter
- NumPy
- SymPy

## Installation

### 1. Install Python

Make sure Python 3.10 or later is installed.

### 2. Install Dependencies

Open PowerShell in the project directory and run:

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Application

Execute the following command:

```bash
python gui.py
```

## Requirements

- Python 3.10 or later
- Tkinter (included with most standard Python installations)
- NumPy 2.1.0 or later
- SymPy 1.13.0 or later

## How to Use

1. Run the application using `python gui.py`.
2. Select the required problem category.
3. Choose the numerical method when applicable.
4. Enter the required mathematical inputs.
5. Click the appropriate Run or Solve button.
6. Review the calculation steps, iteration tables, and final results.

## Important Notes

- The Fixed Point Iteration implementation uses a predefined iteration function: `g(x) = sqrt(1.8x + 2.5)`. It does not automatically derive this function from the entered equation.
- The Golden Section Search implementation uses a maximum of 8 iterations and a tolerance of `0.1`.

## Project Purpose

This project was developed to demonstrate the practical implementation of numerical analysis algorithms through an interactive desktop application.

It provides an educational environment for solving mathematical problems and observing the numerical steps used to obtain solutions.

**This project is intended for educational purposes.**