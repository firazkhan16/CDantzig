# import pandas as pd
# import numpy as np
# from docplex.mp.model import Model
#
# # Load your website advertising data
# page_view_matrix = pd.read_csv("walkthrough/Page_View_Matrix_Example.csv", header=0, index_col=0)
#
# # Compute sample covariance matrix S from page_view_matrix
# # For example, use all data or a subset if needed
# S = page_view_matrix.cov(ddof=1).values
# p = S.shape[0]
#
# # Identity matrix
# I = np.eye(p)
#
# # Set lambda parameter (tuning parameter)
# lambda_value = 0.1
#
# # Create a CPLEX model
# model = Model(name="SparsePrecisionMatrixCDE")
#
# # Decision variables:
# # Theta_{k,j} for k,j in {0,...,p-1}
# # We also need auxiliary variables U_{k,j} to handle the L1 norm of Theta
# Theta = [[model.continuous_var(lb=-model.infinity, name=f"Theta_{k}_{j}") for j in range(p)] for k in range(p)]
# U = [[model.continuous_var(lb=0, name=f"U_{k}_{j}") for j in range(p)] for k in range(p)]
#
# # Objective: minimize sum of U_{k,j} which represents the L1 norm of Theta
# # subject to U_{k,j} >= Theta_{k,j} and U_{k,j} >= -Theta_{k,j}
# for k in range(p):
#     for j in range(p):
#         model.add_constraint(U[k][j] >= Theta[k][j])
#         model.add_constraint(U[k][j] >= -Theta[k][j])
#
#         if k < j:
#             model.add_constraint(Theta[k][j] == Theta[j][k])
#
# model.minimize(model.sum(U[k][j] for k in range(p) for j in range(p)))
#
# # Infinity norm constraints:
# # For each i,j in {0,...,p-1}:
# # -lambda_value <= sum_k S[i,k]*Theta[k,j] - I[i,j] <= lambda_value
# # Note: I[i,j] is 1 if i=j else 0
#
# for i in range(p):
#     print(f"i is {i}")
#     for j in range(p):
#         # Construct linear expression sum_k S[i,k]*Theta[k,j]
#         expr = model.sum(S[i, k] * Theta[k][j] for k in range(p))
#
#         # Add constraints:
#         # expr - I[i,j] <= lambda_value  and  expr - I[i,j] >= -lambda_value
#         model.add_constraint(expr - I[i, j] <= lambda_value)
#         model.add_constraint(expr - I[i, j] >= -lambda_value)
#
# # Solve the model
# solution = model.solve(log_output=True)
# if solution:
#     theta_values = np.array([[solution.get_value(Theta[k][j]) for j in range(p)] for k in range(p)])
#     print("Optimal precision matrix estimate found.")
# else:
#     print("No feasible solution found.")
#
# theta_values now contains the estimated precision matrix Theta.
# Depending on lambda_value, you should see sparse entries in theta_values.


import numpy as np
from docplex.mp.model import Model
import pandas as pd

def cde_columnwise_precision(S, lambda_value=0.1, symmetrize=True, verbose=False):
    """
    Columnwise Constrained Dantzig Estimation of a sparse precision matrix.

    Parameters
    ----------
    S : numpy.ndarray, shape (p, p)
        Sample covariance matrix (p x p).
    lambda_value : float
        Tuning parameter controlling the ∞-norm bound.
    symmetrize : bool
        If True, symmetrize the resulting matrix after each column is solved.
    verbose : bool
        If True, prints additional solver information.

    Returns
    -------
    Omega_est : numpy.ndarray, shape (p, p)
        Estimated precision matrix, where columns are computed one-by-one
        via Constrained Dantzig.  If symmetrize=True, it is forced symmetric.
    """

    p = S.shape[0]
    Omega_est = np.zeros((p, p))  # will store each column in turn

    # For each column i of the precision matrix
    for i in range(p):
        if verbose:
            print(f"Solving for column {i+1}/{p} ...")

        # Build a docplex model for column i
        model = Model(name=f"CDE_col_{i}")
        # Optionally, disable output:
        if not verbose:
            model.context.solver.log_output = False

        # Decision variable w \in R^p, representing beta in the snippet
        w_vars = model.continuous_var_list(p, lb=-model.infinity, name="w")

        # Objective: minimize L1 norm => sum of |w_j|
        # docplex allows model.abs(w_vars[j]) in the expression
        obj_expr = model.sum(model.abs(w_vars[j]) for j in range(p))
        model.minimize(obj_expr)

        # Add ∞-norm constraints: -lambda <= (S[row,:] dot w) - e_i[row] <= lambda
        # Here e_i[row] = 1 if row == i else 0
        for row in range(p):
            e_val = 1.0 if (row == i) else 0.0
            dot_expr = model.sum(S[row, col] * w_vars[col] for col in range(p))
            # dot_expr - e_val <= lambda_value
            model.add_constraint(dot_expr - e_val <= lambda_value)
            # dot_expr - e_val >= -lambda_value
            model.add_constraint(dot_expr - e_val >= -lambda_value)

        # Solve the model
        solution = model.solve()
        if solution is None:
            # Infeasible or no solution found
            raise RuntimeError(f"No solution found for column {i}. Possibly increase lambda_value.")

        # Retrieve solution values and store in Omega_est
        w_values = [solution.get_value(wv) for wv in w_vars]
        for row in range(p):
            Omega_est[row, i] = w_values[row]

        # Clear the model
        model.clear()

    # Optional: force symmetry
    if symmetrize:
        # A simple approach: average upper and lower triangles
        for i in range(p):
            for j in range(i+1, p):
                val = 0.5 * (Omega_est[i, j] + Omega_est[j, i])
                Omega_est[i, j] = val
                Omega_est[j, i] = val

    return Omega_est

if __name__ == "__main__":
    page_view_matrix = pd.read_csv("walkthrough/Page_View_Matrix_Example.csv", header=0, index_col=0)
    S = page_view_matrix.cov(ddof=1).values
    p = S.shape[0]

    # Dantzig-based columnwise estimation
    lambda_val = 0.5
    Omega_est = cde_columnwise_precision(S, lambda_val, symmetrize=True, verbose=True)

    print("Estimated Precision Matrix (Columnwise Dantzig):")
    print(Omega_est)