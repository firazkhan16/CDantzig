import numpy as np
import pandas as pd
from docplex.mp.model import Model


def cde_column(S, e_i, lambda_value, fixed_values=None):
    """
    Solve one column of Constrained Dantzig for sparse precision estimation,
    optionally fixing some entries to previously determined values (beta_j).

    Parameters
    ----------
    S : np.ndarray, shape (p, p)
        Sample covariance matrix (p x p).
    e_i : np.ndarray, shape (p,)
        Standard basis vector with 1 in the i-th position, 0 otherwise.
    lambda_value : float
        Tuning parameter for infinity norm.
    fixed_values : dict or None
        A dictionary of {index -> value} for entries in the solution that must be fixed.
        E.g. if fixed_values = {0: 0.1, 3: -0.05}, then beta_0=0.1, beta_3=-0.05.

    Returns
    -------
    beta : np.ndarray, shape (p,)
        The solution vector to the subproblem.
    """
    p = S.shape[0]
    model = Model(name="CDE_col")
    model.context.solver.log_output = False  # set True for debug logs

    # Decision variables: w_j for j=0..p-1
    w_vars = model.continuous_var_list(p, lb=-model.infinity, name="beta")

    # Objective: minimize sum of absolute values => L1 norm
    obj_expr = model.sum(model.abs(w_vars[j]) for j in range(p))
    model.minimize(obj_expr)

    # Infinity norm constraints: -lambda <= (S[row,:] dot w) - e_i[row] <= lambda
    for row in range(p):
        lhs = model.sum(S[row, col] * w_vars[col] for col in range(p)) - e_i[row]
        model.add_constraint(lhs <= lambda_value)
        model.add_constraint(lhs >= -lambda_value)

    # If some entries are fixed, set w_vars[j] = that fixed value
    if fixed_values is not None:
        for j, val in fixed_values.items():
            model.add_constraint(w_vars[j] == val)

    # Solve
    solution = model.solve()
    if solution is None:
        model.clear()
        raise RuntimeError("No feasible solution found. Possibly increase lambda_value.")

    beta = np.array([solution.get_value(wv) for wv in w_vars], dtype=float)
    model.clear()
    return beta


def cde_columnwise_recursive(S):
    """
    Recursive/Sequential approach to estimate columns (Omega_{:,i}) one-by-one
    while referencing previously solved columns for partial consistency:
       beta_j = Omega_{i,j}, for j < i.

    This implements the idea from the snippet:
      Omega_hat_i = argmin ||beta||_1
         subject to  ||S_n beta - e_i||_∞ <= lambda
         and  beta_j = Omega_hat_{i,j}   for 1 <= j < i.

    Parameters
    ----------
    S : np.ndarray, shape (p, p)
        Sample covariance matrix.
    lambda_value : float
        Tuning parameter for the infinity norm constraints.

    Returns
    -------
    Omega_hat : np.ndarray, shape (p, p)
        The resulting matrix from the iterative approach.
    """
    p = S.shape[0]
    Omega_hat = np.zeros((p, p), dtype=float)

    for i in range(p):
        print(f"Starting Column {i}")
        # i-th basis vector
        e_i = np.zeros(p)
        e_i[i] = 1.0


        fixed_dict = {}
        for j in range(i):
            fixed_dict[j] = Omega_hat[i, j]  # keep same (i, j) as row i from col j

        # Solve for column i with partial constraints
        beta_i = cde_column(S, e_i, 0.9, fixed_values=fixed_dict)

        for row in range(p):
            Omega_hat[row, i] = beta_i[row]

    return Omega_hat


if __name__ == "__main__":
    page_view_matrix = pd.read_csv("walkthrough/Page_View_Matrix_Example.csv", header=0, index_col=0)
    S = page_view_matrix.cov(ddof=1).values

    # Omega_est = cde_columnwise_recursive(S)
    # np.save("Omega_est.npy", Omega_est)

    loaded_Omega = np.load("Omega_est.npy")