import pandas as pd
import numpy as np
from docplex.mp.model import Model


def find_lambda_max_cplex(sigma, eta, A, b, p, k, factor=1.0):
    """
    Compute lambda_max and a scaling factor for the given CDE problem setup.

    Parameters:
    - sigma: Covariance matrix (p x p)
    - eta: Mean vector of dimension p
    - A: Constraint matrix (k x p)
    - b: Constraint vector (k x 1)
    - p: Dimension of w (beta)
    - k: Dimension of gamma
    - factor: A scaling factor (default 1.0)

    Returns:
    - lambda_max: The smallest feasible lambda that satisfies the constraints.
    - new_factor: A scaling factor so that lambda_max * new_factor = 1.
                  If lambda_max = 0, returns the original factor.
    """

    # PHASE 1: Solve lp1 to find minimal L1 norm solution satisfying A w = b
    model = Model("lp1")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    # Define w variables
    w = np.array(model.continuous_var_list(p, lb=-model.infinity, name="w"))

    # Objective: minimize sum of abs(w[i])
    # model.abs() is a docplex feature that linearizes absolute values internally.
    model.minimize(model.sum(model.abs(w[i]) for i in range(p)))

    # Add linear equality constraints A w = b
    for i in range(k):
        expr = model.dot(w, A[i])
        model.add_constraint(expr == b[i])

    # Solve lp1
    solution = model.solve()
    if solution is None or model.solve_status.value != 2:
        model.clear()
        raise Exception(
            "Infeasible lp1: Could not find a feasible solution to the baseline problem."
        )

    lp1_norm = solution.get_objective_value()
    model.clear()

    # PHASE 2: Solve lp2 to find minimal lambda
    model = Model("lp2")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    w = np.array(model.continuous_var_list(p, lb=-model.infinity, name="w"))
    gamma = np.array(model.continuous_var_list(k, lb=-model.infinity, name="gamma"))
    _lambda_scaled = model.continuous_var(name="lambda")

    # Minimize lambda
    model.minimize(_lambda_scaled)

    # Infinity norm constraints:
    # For each i in [0, p-1], we have:
    # -_lambda_scaled <= factor*(sigma[i]*w - eta[i] + (A.T[i]*gamma)) <= _lambda_scaled
    # which we split into two constraints each.
    for i in range(p):
        expr = (
            factor
            * (model.dot(w, sigma.iloc[i, :]) - eta[i] + model.dot(gamma, A.T[i]))
            - _lambda_scaled
        )
        model.add_constraint(expr <= 0)

        expr = (
            factor
            * (model.dot(w, sigma.iloc[i, :]) - eta[i] + model.dot(gamma, A.T[i]))
            + _lambda_scaled
        )
        model.add_constraint(expr >= 0)

    # A w = b constraints again
    for i in range(k):
        expr = model.dot(w, A[i])
        model.add_constraint(expr == b[i])

    # Enforce the same L1 norm as lp1_norm
    expr = model.sum(model.abs(w[i]) for i in range(p))
    model.add_constraint(expr == lp1_norm)

    # Solve lp2
    solution = model.solve()
    if solution is None or model.solve_status.value != 2:
        model.clear()
        raise Exception("Infeasible lp2: Could not find a feasible lambda.")

    lambda_max = solution.get_value(_lambda_scaled)
    if lambda_max != 0:
        new_factor = factor / lambda_max
    else:
        new_factor = factor

    model.clear()
    return lambda_max, new_factor


def CDE_DOcplex(sigma, eta, A, b, p, k, factor, _lambda_scaled):
    """
    Compute minimum beta subject to linear constraints.

    Parameters:
    - sigma: Covariance matrix (p x p)
    - eta: Mean vector of dimension p
    - A: Constraint matrix (k x p)
    - b: Constraint vector (k x 1)
    - p: Dimension of w (beta)
    - k: Dimension of gamma
    - factor: A scaling factor (default 1.0)
    - _lambda_scaled: tuning parameter

    Returns:
    - w: the set of optimised weights
    """

    model = Model(name="CDE")

    # From Dechuan's reference code
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    # Define variables similar to reference code: w and gamma
    w = np.array(
        model.continuous_var_list([f"w{i}" for i in range(p)], lb=-model.infinity)
    )
    gamma = np.array(
        model.continuous_var_list([f"gamma{i}" for i in range(k)], lb=-model.infinity)
    )

    # Objective: minimize the L1 norm of w using model.abs()
    expr = model.sum(model.abs(w[i]) for i in range(p))
    model.minimize(expr)

    # Infinity norm constraints:
    for row in range(p):
        expr = (
            factor
            * (model.dot(w, sigma.values[row]) - eta[row] + model.dot(gamma, A.T[row]))
            - _lambda_scaled
        )
        model.add_constraint(expr <= 0)

        expr = (
            factor
            * (model.dot(w, sigma.values[row]) - eta[row] + model.dot(gamma, A.T[row]))
            + _lambda_scaled
        )
        model.add_constraint(expr >= 0)

    # Equality constraints A beta = b
    for i in range(k):
        expr = model.dot(w, A[i])
        model.add_constraint(expr == b[i])

    # Not needed if A is set to row vector of ones
    # model.add_constraint(model.sum(w[i] for i in range(p)) == 1)

    solution = model.solve()
    if model.solve_status.value == 2:
        optimised_w = np.array(solution.get_values(w))
        model.clear()
        return optimised_w
    else:
        model.clear()
        return "Error converging to a solution"


def constrained_lasso(X, eta, A, b, p, k, lambda_value):
    """
    Compute beta subject to Constrained Lasso conditions.

    Parameters:
    - X: Design matrix (n x p)
    - eta: Some n dimensional target vector
    - A: Constraint matrix (k x p)
    - b: Constraint vector (k x 1)
    - p: Dimension of w (beta)
    - k: Dimension of gamma
    - _lambda_scaled: tuning parameter

    Returns:
    - w: the set of optimised weights
    """
    model = Model("constrained_lasso")

    w = model.continuous_var_list(p, name="w", lb=0)
    n = len(eta)

    residual = [model.dot(w, X.values[i]) - eta[i] for i in range(n)]

    quad_obj = model.sum(residual[i] * residual[i] for i in range(n)) * 0.5

    l1_penalty = lambda_value * model.sum(model.abs(w[i]) for i in range(p))

    model.minimize(quad_obj + l1_penalty)

    for i in range(k):
        model.add_constraint(model.dot(w, A[i]) == b[i])

    solution = model.solve()
    if solution is not None:
        w_values = [solution.get_value(var) for var in w]
        model.clear()
        return np.array(w_values)
    else:
        model.clear()
        return "Error converging to a solution"


def calculate_metrics(Z, w, q):
    """
    Calculate reach and CTR

    Parameters:
    - Z (numpy.ndarray): Page view matrix (n x p), where n is the number of users, p is the number of websites.
    - w (numpy.ndarray): Weights (budget allocations) of length p.
    - q (numpy.ndarray): Clickthrough rates for each website (length p).

    Returns:
    - reach (float): Fraction of users reached.
    - ctr (float): Click-through rate.
    """

    # Total exposure per user
    exposure = Z.values @ w  # Matrix-vector multiplication (n-dimensional result)

    # Reach: Fraction of users with exposure > 0
    num_users = Z.shape[0]
    reach = np.sum(exposure > 0) / num_users

    # Clicks: Exposure weighted by clickthrough rates
    clicks = Z.values @ (w * q)  # Element-wise multiplication of weights and CTRs

    # CTR: Fraction of users with at least one click
    ctr = np.sum(clicks > 0) / num_users

    return reach, ctr


def compute_reach_and_ctr(X1, Y1, beta):
    """
    Computes Reach and CTR based on page views, site information, and budget allocation.

    Parameters:
    - page_view_matrix (pd.DataFrame): Rows are machines, columns are websites, values are page views (z_ij).
    - site_info (pd.DataFrame): Contains columns ['Site_Name', 'Cost', 'Pages', 'Clickthrough'].
    - beta (pd.Series): Budget allocation weights, indexed by Site_Name.

    Returns:
    - reach (float): Fraction of users exposed to at least one ad.
    - ctr (float): Fraction of users who clicked on at least one ad.
    """
    X = X1.copy()
    Y = Y1.copy()
    # Step 1: Compute gamma (fraction of ads per dollar spent)
    Y = Y.set_index("Site_Name")  # Set Site_Name as index for alignment
    Y["gamma"] = 1 / ((Y["Pages"] / 1000) * Y["Cost"])

    # Step 2: Align beta with gamma using index
    gamma_series = Y["gamma"]
    scaling_factors = (
        beta * gamma_series
    )  # Element-wise multiplication to get beta * gamma

    # Step 3: Adjust page_view_matrix with scaling factors
    adjusted_matrix = X.multiply(scaling_factors, axis=1)

    # Step 4: Compute Reach
    reach_matrix = 1 - adjusted_matrix  # (1 - beta_j * gamma_j) for each website
    reach_matrix = reach_matrix.pow(X)  # (1 - beta_j * gamma_j)^z_ij
    reach_per_user = reach_matrix.prod(axis=1)  # Product across websites for each user
    reach = 1 - reach_per_user.mean()  # Average across users and subtract from 1

    # Step 5: Compute CTR
    ctr_matrix = 1 - adjusted_matrix.multiply(
        Y["Clickthrough"], axis=1
    )  # (1 - beta_j * gamma_j * q_j)
    ctr_matrix = ctr_matrix.pow(X)  # (1 - beta_j * gamma_j * q_j)^z_ij
    ctr_per_user = ctr_matrix.prod(axis=1)  # Product across websites for each user
    ctr = 1 - ctr_per_user.mean()  # Average across users and subtract from 1

    return reach, ctr


if __name__ == "__main__":
    # TODO: We have yet to consider demographic constraints

    page_view_matrix = pd.read_csv(
        r"walkthrough/Page_View_Matrix_Example.csv", header=0, index_col=0
    )
    site_info = pd.read_csv(r"walkthrough/500_Site_Info_Example.csv", header=0)

    sigma = page_view_matrix.iloc[:-1, :].cov(ddof=1)
    eta = page_view_matrix.iloc[:-1, :].mean().values

    p = 500
    k = 1
    factor = 1.0
    lambda_value = 0.1

    result_dic = {}
    portfolio_dic = {}
    lambda_list = [i / 20 for i in range(19, 0, -1)]
    for _lambda in lambda_list:
        portfolio_dic[_lambda] = []

    A = np.ones(
        (1, p)
    )  # A represents a row vector of ones so A*beta = b represents sum of beta = b
    b = np.array([1.0])  # b represents total website budget

    lambda_max, original_factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    if lambda_max <= 0:
        raise Exception("Lambda_max=0!")

    return_list = np.empty(len(lambda_list))
    for index, _lambda in enumerate(lambda_list):
        print(f"Optimizing {index}:{_lambda}")
        w = CDE_DOcplex(sigma, eta, A, b, p, k, original_factor, _lambda)

        portfolio_dic[_lambda].append(w)
        return_list[index] = page_view_matrix.iloc[-1].values @ w
    result_dic[0] = return_list
    result = pd.DataFrame.from_dict(result_dic)
    result.index = lambda_list
    result.to_csv(f"results.csv")
    portfolios = pd.DataFrame.from_dict(portfolio_dic, orient="index")
    portfolios.to_csv(f"portfolios.csv")
    print("Done with CDE")

    p = 500
    k = 1
    A = np.ones((1, p))
    b = np.array([100])
    lambda_value = 0.1

    eta = page_view_matrix.sum(axis=1).values

    w = constrained_lasso(page_view_matrix, eta, A, b, p, k, lambda_value)

    # reach, ctr = calculate_metrics(page_view_matrix, w, site_info["Clickthrough"])

    reach, ctr = compute_reach_and_ctr(page_view_matrix, site_info, w)

    print("Done with CLasso")
