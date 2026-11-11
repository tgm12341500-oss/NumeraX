import math

class GoldenSectionSolver:
    def __init__(self, func, xl, xu):
        self.func = func
        self.xl = xl
        self.xu = xu
        self.steps = []
        self.initial_interval = abs(xu - xl)

    def golden_section_search(self, maximize=True):
        """Golden Section Search to find the extremum (max or min)."""
        gr = 0.618  # Use the rounded golden ratio to match the example
        xl, xu = self.xl, self.xu
        iteration_data = []
        iteration = 0
        max_iter = 8  # Match the example: stop after 8 iterations
        tol = 0.1  # Tolerance for interval size
        epsilon = 1e-8  # Tolerance for comparing function values

        # Validate initial interval
        if xl >= xu:
            self.steps.append("Error: xl must be less than xu.")
            return iteration_data, None

        # Initial setup
        d = gr * (xu - xl)
        x1 = xl + d
        x2 = xu - d
        f_xl = self.func(xl)
        f_xu = self.func(xu)
        f_x1 = self.func(x1)
        f_x2 = self.func(x2)

        # Check if interval brackets an extremum
        if maximize:
            if f_x1 <= f_xl + epsilon or f_x2 <= f_xu + epsilon:
                self.steps.append("Error: xl and xu do not bracket a maximum. Please choose a different interval.")
                return iteration_data, None
        else:
            if f_x1 >= f_xl - epsilon or f_x2 >= f_xu - epsilon:
                self.steps.append("Error: xl and xu do not bracket a minimum. Please choose a different interval.")
                return iteration_data, None

        while iteration < max_iter:
            # Log iteration data in the format expected by the GUI
            iteration_data.append([
                iteration + 1,
                f"{xl:.3f}",
                f"{f_xl:.3f}" if f_xl is not None else "-",
                f"{x2:.3f}",
                f"{f_x2:.3f}" if f_x2 is not None else "-",
                f"{x1:.3f}",
                f"{f_x1:.3f}" if f_x1 is not None else "-",
                f"{xu:.3f}",
                f"{f_xu:.3f}" if f_xu is not None else "-",
                f"{d:.3f}"
            ])
            self.steps.append(
                f"Iteration {iteration + 1} ({'Maximizing' if maximize else 'Minimizing'}): "
                f"xl={xl:.3f}, f(xl)={f_xl:.3f}, x2={x2:.3f}, f(x2)={f_x2:.3f}, "
                f"x1={x1:.3f}, f(x1)={f_x1:.3f}, xu={xu:.3f}, f(xu)={f_xu:.3f}, d={d:.3f}"
            )

            # Stopping condition: interval size (only after minimum iterations)
            if iteration >= max_iter - 1 or xu - xl < tol:
                value = x2 if (f_x2 > f_x1) == maximize else x1
                f_value = self.func(value)
                self.steps.append(
                    f"{'Maximum' if maximize else 'Minimum'} at x = {value:.4f}, f(x) = {f_value:.4f}"
                )
                return iteration_data, value

            # Update interval based on function values (corrected logic)
            if maximize:
                if f_x1 > f_x2 + epsilon:
                    # Maximum in [x2, xu]
                    xl = x2
                    f_xl = f_x2
                    x2 = x1
                    f_x2 = f_x1
                    d = gr * (xu - xl)
                    x1 = xl + d
                    f_x1 = self.func(x1)
                    self.steps.append(f"Choosing [x2, xu]: new xl = {xl:.3f}, new x1 = {x1:.3f}")
                else:
                    # Maximum in [xl, x1]
                    xu = x1
                    f_xu = f_x1
                    x1 = x2
                    f_x1 = f_x2
                    d = gr * (xu - xl)
                    x2 = xu - d
                    f_x2 = self.func(x2)
                    self.steps.append(f"Choosing [xl, x1]: new xu = {xu:.3f}, new x2 = {x2:.3f}")
            else:
                if f_x1 < f_x2 - epsilon:
                    # Minimum in [x2, xu]
                    xl = x2
                    f_xl = f_x2
                    x2 = x1
                    f_x2 = f_x1
                    d = gr * (xu - xl)
                    x1 = xl + d
                    f_x1 = self.func(x1)
                    self.steps.append(f"Choosing [x2, xu]: new xl = {xl:.3f}, new x1 = {x1:.3f}")
                else:
                    # Minimum in [xl, x1]
                    xu = x1
                    f_xu = f_x1
                    x1 = x2
                    f_x1 = f_x2
                    d = gr * (xu - xl)
                    x2 = xu - d
                    f_x2 = self.func(x2)
                    self.steps.append(f"Choosing [xl, x1]: new xu = {xu:.3f}, new x2 = {x2:.3f}")

            iteration += 1

        # If max iterations reached
        value = x2 if (f_x2 > f_x1) == maximize else x1
        f_value = self.func(value)
        self.steps.append(
            f"{'Maximum' if maximize else 'Minimum'} at x = {value:.4f}, f(x) = {f_value:.4f}"
        )
        return iteration_data, value