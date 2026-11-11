import numpy as np

class MatrixSolver:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)
        self.n = len(self.matrix)
        self.steps = []

    def check_singular(self, A):
        """Check if matrix is singular by computing determinant."""
        det = np.linalg.det(A)
        if np.isclose(det, 0):
            return True, "Matrix is singular (determinant ≈ 0). No unique solution exists."
        return False, ""

    def gauss_elimination(self):
        """Gauss Elimination without pivoting"""
        a = self.matrix.copy()
        self.steps = [f"Initial Matrix:\n{a}"]
        
        # Separate coefficients (A) and constants (b)
        A = a[:, :-1]
        b = a[:, -1]

        # Check if matrix is singular
        is_singular, message = self.check_singular(A)
        if is_singular:
            self.steps.append(message)
            return None, self.steps

        # Forward elimination
        for i in range(self.n):
            if abs(a[i, i]) < 1e-10:
                self.steps.append("Error: Pivot too small - Matrix may be singular")
                return None, self.steps
            for j in range(i + 1, self.n):
                factor = a[j, i] / a[i, i]
                a[j, :] -= factor * a[i, :]
                self.steps.append(f"Eliminate Row {j+1} using Row {i+1} (factor: {factor:.6f}):\n{a}")
        
        # Back substitution
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = (a[i, -1] - np.dot(a[i, i:self.n], x[i:self.n])) / a[i, i]
        
        self.steps.append(f"Solution vector:\n{x}")
        return x, self.steps

    def gauss_pivoting(self):
        """Gauss Elimination with Partial Pivoting"""
        a = self.matrix.copy()
        self.steps = [f"Initial Matrix:\n{a}"]
        
        # Separate coefficients (A) and constants (b)
        A = a[:, :-1]
        b = a[:, -1]

        # Check if matrix is singular
        is_singular, message = self.check_singular(A)
       
        if is_singular:
            self.steps.append(message)
            return None, self.steps

        # Forward elimination with partial pivoting
        for i in range(self.n):
            # Find pivot
            pivot_row = i + np.argmax(np.abs(a[i:, i]))
            if abs(a[pivot_row, i]) < 1e-10:
                self.steps.append("Error: Pivot too small - Matrix may be singular")
                return None, self.steps
            if pivot_row != i:
                a[[i, pivot_row]] = a[[pivot_row, i]]
                self.steps.append(f"Swap Row {i+1} with Row {pivot_row+1}:\n{a}")
            
            for j in range(i + 1, self.n):
                factor = a[j, i] / a[i, i]
                a[j, :] -= factor * a[i, :]
                self.steps.append(f"Eliminate Row {j+1} using Row {i+1} (factor: {factor:.6f}):\n{a}")
        
        # Back substitution
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = (a[i, -1] - np.dot(a[i, i:self.n], x[i:self.n])) / a[i, i]
        
        self.steps.append(f"Solution vector:\n{x}")
        return x, self.steps

    def gauss_jordan(self):
        """Gauss-Jordan Elimination without pivoting"""
        a = self.matrix.copy()
        self.steps = [f"Initial Matrix:\n{a}"]
        
        # Separate coefficients (A) and constants (b)
        A = a[:, :-1]
        b = a[:, -1]

        # Check if matrix is singular
        is_singular, message = self.check_singular(A)
        if is_singular:
            self.steps.append(message)
            return None, self.steps

        # Forward elimination to get diagonal matrix
        for i in range(self.n):
            if abs(a[i, i]) < 1e-10:
                self.steps.append("Error: Pivot too small - Matrix may be singular")
                return None, self.steps
            a[i, :] = a[i, :] / a[i, i]
            self.steps.append(f"Normalize Row {i+1}:\n{a}")
            for j in range(self.n):
                if j != i:
                    factor = a[j, i]
                    a[j, :] -= factor * a[i, :]
                    self.steps.append(f"Eliminate Row {j+1} using Row {i+1} (factor: {factor:.6f}):\n{a}")
        
        x = a[:, -1]
        self.steps.append(f"Solution vector:\n{x}")
        return x, self.steps

    def gauss_jordan_pivoting(self):
        """Gauss-Jordan Elimination with Partial Pivoting"""
        a = self.matrix.copy()
        self.steps = [f"Initial Matrix:\n{a}"]
        
        # Separate coefficients (A) and constants (b)
        A = a[:, :-1]
        b = a[:, -1]

        # Check if matrix is singular
        is_singular, message = self.check_singular(A)
        if is_singular:
            self.steps.append(message)
            return None, self.steps

        # Forward elimination with partial pivoting
        for i in range(self.n):
            pivot_row = i + np.argmax(np.abs(a[i:, i]))
            if abs(a[pivot_row, i]) < 1e-10:
                self.steps.append("Error: Pivot too small - Matrix may be singular")
                return None, self.steps
            if pivot_row != i:
                a[[i, pivot_row]] = a[[pivot_row, i]]
                self.steps.append(f"Swap Row {i+1} with Row {pivot_row+1}:\n{a}")
            
            a[i, :] = a[i, :] / a[i, i]
            self.steps.append(f"Normalize Row {i+1}:\n{a}")
            for j in range(self.n):
                if j != i:
                    factor = a[j, i]
                    a[j, :] -= factor * a[i, :]
                    self.steps.append(f"Eliminate Row {j+1} using Row {i+1} (factor: {factor:.6f}):\n{a}")
        
        x = a[:, -1]
        self.steps.append(f"Solution vector:\n{x}")
        return x, self.steps

    def lu_decomposition(self):
        """LU Decomposition without pivoting"""
        a = self.matrix.copy()
        self.steps = [f"Initial Matrix:\n{a}"]
        
        # Separate coefficients (A) and constants (b)
        A = a[:, :-1]
        b = a[:, -1]

        # Check if matrix is singular
        is_singular, message = self.check_singular(A)
        if is_singular:
            self.steps.append(message)
            return None, self.steps

        L = np.eye(self.n)
        U = a[:, :-1].copy()

        # LU decomposition
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if abs(U[i, i]) < 1e-10:
                    self.steps.append("Error: Pivot too small - Matrix may be singular")
                    return None, self.steps
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j, :] -= factor * U[i, :]
                self.steps.append(f"Update L[{j+1},{i+1}] = {factor:.6f}, U:\n{U}")

        self.steps.append(f"L Matrix:\n{L}")
        self.steps.append(f"U Matrix:\n{U}")

        # Forward substitution: Ly = b
        y = np.zeros(self.n)
        for i in range(self.n):
            y[i] = b[i] - np.dot(L[i, :i], y[:i])
        
        # Backward substitution: Ux = y
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
        self.steps.append(f"Solution vector:\n{x}")
        return x, self.steps

    def lu_pivoting(self):
        """LU Decomposition with Partial Pivoting"""
        a = self.matrix.copy()
        self.steps = [f"Initial Matrix:\n{a}"]
        
        # Separate coefficients (A) and constants (b)
        A = a[:, :-1]
        b = a[:, -1]

        # Check if matrix is singular
        is_singular, message = self.check_singular(A)
        if is_singular:
            self.steps.append(message)
            return None, self.steps

        P = np.eye(self.n)
        L = np.eye(self.n)
        U = a[:, :-1].copy()

        # LU decomposition with partial pivoting
        for i in range(self.n):
            pivot_row = i + np.argmax(np.abs(U[i:, i]))
            if abs(U[pivot_row, i]) < 1e-10:
                self.steps.append("Error: Pivot too small - Matrix may be singular")
                return None, self.steps
            if pivot_row != i:
                U[[i, pivot_row]] = U[[pivot_row, i]]
                P[[i, pivot_row]] = P[[pivot_row, i]]
                L[[i, pivot_row], :i] = L[[pivot_row, i], :i]
                self.steps.append(f"Swap Row {i+1} with Row {pivot_row+1}:\n{U}")
            
            for j in range(i + 1, self.n):
                factor = U[j, i] / U[i, i]
                L[j, i] = factor
                U[j, :] -= factor * U[i, :]
                self.steps.append(f"Update L[{j+1},{i+1}] = {factor:.6f}, U:\n{U}")

        self.steps.append(f"P Matrix:\n{P}")
        self.steps.append(f"L Matrix:\n{L}")
        self.steps.append(f"U Matrix:\n{U}")

        # Solve: PAx = Pb => LUx = Pb
        Pb = np.dot(P, b)
        
        # Forward substitution: Ly = Pb
        y = np.zeros(self.n)
        for i in range(self.n):
            y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
        
        # Backward substitution: Ux = y
        x = np.zeros(self.n)
        for i in range(self.n - 1, -1, -1):
            x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
        self.steps.append(f"Solution vector:\n{x}")
        return x, self.steps

    def cramers_rule(self):
        """Cramer's Rule"""
        A = self.matrix[:, :-1].copy()
        b = self.matrix[:, -1].copy()
        self.steps = []
        
        # Check if matrix is singular
        detA = np.linalg.det(A)
        self.steps.append(f"Determinant of A: {detA:.6f}")
        
        if np.isclose(detA, 0):
            self.steps.append("Determinant is zero - No unique solution exists")
            return None, self.steps
        
        x = np.zeros(self.n)
        for i in range(self.n):
            Ai = A.copy()
            Ai[:, i] = b
            detAi = np.linalg.det(Ai)
            x[i] = detAi / detA
            self.steps.append(f"Determinant of A{i+1}: {detAi:.6f}, x{i+1} = {x[i]:.6f}")
        
        self.steps.append(f"Solution vector:\n{x}")
        return x, self.steps