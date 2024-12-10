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
    model = Model('lp1')
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    # Define w variables
    w = np.array(model.continuous_var_list(p, lb=-model.infinity, name='w'))

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
        raise Exception('Infeasible lp1: Could not find a feasible solution to the baseline problem.')

    lp1_norm = solution.get_objective_value()
    model.clear()

    # PHASE 2: Solve lp2 to find minimal lambda
    model = Model('lp2')
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    w = np.array(model.continuous_var_list(p, lb=-model.infinity, name='w'))
    gamma = np.array(model.continuous_var_list(k, lb=-model.infinity, name='gamma'))
    _lambda_scaled = model.continuous_var(name='lambda')

    # Minimize lambda
    model.minimize(_lambda_scaled)

    # Infinity norm constraints:
    # For each i in [0, p-1], we have:
    # -_lambda_scaled <= factor*(sigma[i]*w - eta[i] + (A.T[i]*gamma)) <= _lambda_scaled
    # which we split into two constraints each.
    for i in range(p):
        expr = factor * (model.dot(w, sigma.iloc[i,:]) - eta[i] + model.dot(gamma, A.T[i])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (model.dot(w, sigma.iloc[i,:]) - eta[i] + model.dot(gamma, A.T[i])) + _lambda_scaled
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
        raise Exception('Infeasible lp2: Could not find a feasible lambda.')

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

    model = Model(name='CDE')

    # From Dechuan's reference code
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    # Define variables similar to reference code: w and gamma
    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))

    # Objective: minimize the L1 norm of w using model.abs()
    expr = model.sum(model.abs(w[i]) for i in range(p))
    model.minimize(expr)

    # Infinity norm constraints:
    for row in range(p):
        expr = factor * (model.dot(w, sigma.values[row]) - eta[row] + model.dot(gamma, A.T[row])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (model.dot(w, sigma.values[row]) - eta[row] + model.dot(gamma, A.T[row])) + _lambda_scaled
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
        return 'Error converging to a solution'


if __name__ == '__main__':

    page_view_matrix = pd.read_csv(r"walkthrough/Page_View_Matrix_Example.csv", header=0, index_col=0)
    site_info = pd.read_csv(r"walkthrough/500_Site_Info_Example.csv", header=0)

    sigma = page_view_matrix.cov(ddof=1)
    eta = page_view_matrix.mean().values

    p = 500
    k = 1
    factor = 1.0
    lambda_value = 0.1

    result_dic = {}
    portfolio_dic = {}
    lambda_list = [i / 20 for i in range(19, 0, -1)]
    for _lambda in lambda_list:
        portfolio_dic[_lambda] = []

    A = np.ones((1, p))  # A is a 1x500 matrix, all entries are 1
    b = np.array([1.0])  # b is a 1-dimensional vector with the value 1

    lambda_max, original_factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    if lambda_max<=0:
        raise Exception('Lambda_max=0!')

    return_list = np.empty(len(lambda_list))
    for index, _lambda in enumerate(lambda_list):
        print(f'Optimizing {index}:{_lambda}')
        w = CDE_DOcplex(sigma, eta, A, b, p, k, original_factor, _lambda)

        portfolio_dic[_lambda].append(w)
        # return_list[index] = data.iloc[-1].values @ w

    result = pd.DataFrame.from_dict(result_dic)
    result.index = lambda_list
    portfolios = pd.DataFrame.from_dict(portfolio_dic, orient='index')
    portfolios.to_csv(
        f'website_results.csv'
    )
