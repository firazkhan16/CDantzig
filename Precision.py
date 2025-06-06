import numpy as np
from scipy.stats import multivariate_normal
from docplex.mp.model import Model
from sklearn.model_selection import KFold
import pandas as pd
import time
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def find_lambda_max_cde_v1(S, e_i, p, factor, fixed_values=None):
    model = Model("lp1")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    w = np.array(
        model.continuous_var_list([f"w{i}" for i in range(p)], lb=-model.infinity)
    )

    # Objective: minimize sum of abs(w[i])
    model.minimize(model.sum(model.abs(w[i]) for i in range(p)))

    if fixed_values is not None:
        for j, val in fixed_values.items():
            model.add_constraint(w[j] == val)

    solution = model.solve()
    if solution is None or model.solve_status.value != 2:
        model.clear()
        raise Exception(
            "Infeasible lp1: Could not find a feasible solution to the baseline problem."
        )

    lp1_norm = solution.get_objective_value()
    model.clear()

    # PHASE 2: Solve lp2 to find max lambda
    model = Model("lp2")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    w = np.array(
        model.continuous_var_list([f"w{i}" for i in range(p)], lb=-model.infinity)
    )
    _lambda_scaled = model.continuous_var(name="lambda")

    model.minimize(_lambda_scaled)

    # Infinity norm constraints: -lambda <= (S[row,:] dot w) - e_i[row] <= lambda
    for row in range(p):
        lhs = model.sum(S[row, col] * w[col] for col in range(p)) - e_i[row]
        model.add_constraint(lhs <= _lambda_scaled)
        model.add_constraint(lhs >= -_lambda_scaled)

    if fixed_values is not None:
        for j, val in fixed_values.items():
            model.add_constraint(w[j] == val)

    expr = model.sum(model.abs(w[i]) for i in range(p))
    model.add_constraint(expr == lp1_norm)

    # Solve lp2
    solution = model.solve()
    if solution is None or model.solve_status.value != 2:
        model.clear()
        raise Exception("Infeasible lp2: Could not find a feasible lambda.")

    lambda_max = solution.get_value(_lambda_scaled)
    new_factor = 1 / lambda_max * factor if lambda_max != 0 else factor
    model.clear()
    return lambda_max, new_factor


def find_lambda_min_cde_v1(S, e_i, scaling_factor, fixed_values=None):
    lambda_list = [i / 20 for i in range(1, 20)]
    left = 0
    right = len(lambda_list) - 1
    feasible_lambda = None

    while left <= right:
        mid = (left + right) // 2
        lam = lambda_list[mid]
        try:
            _ = cde_column(S, e_i, lam, scaling_factor, fixed_values)
            feasible_lambda = lam
            right = mid - 1
        except RuntimeError:
            left = mid + 1
    return feasible_lambda if feasible_lambda is not None else 1


def cde_column(S, e_i, _lambda_scaled, scaling_factor, fixed_values=None):
    p = S.shape[0]
    model = Model(name="CDE_col")

    w_vars = model.continuous_var_list(p, lb=-model.infinity, name="beta")

    obj_expr = model.sum(model.abs(w_vars[j]) for j in range(p))
    model.minimize(obj_expr)

    # Infinity norm constraints: -lambda <= (S[row,:] dot w) - e_i[row] <= lambda
    for row in range(p):
        lhs = model.sum(S[row, col] * w_vars[col] for col in range(p)) - e_i[row]
        model.add_constraint(lhs <= _lambda_scaled / scaling_factor)
        model.add_constraint(lhs >= -_lambda_scaled / scaling_factor)

    if fixed_values is not None:
        for j, val in fixed_values.items():
            model.add_constraint(w_vars[j] == val)

    solution = model.solve()
    if solution is None:
        model.clear()
        raise RuntimeError("No feasible solution found for column.")

    beta = np.array([solution.get_value(wv) for wv in w_vars], dtype=float)
    model.clear()
    return beta


def cde_column_v1(S, e_i, _lambda, lambda_max, lambda_min, fixed_values=None):
    p = S.shape[0]
    model = Model(name="CDE_col")

    w_vars = model.continuous_var_list(p, lb=-model.infinity, name="beta")

    obj_expr = model.sum(model.abs(w_vars[j]) for j in range(p))
    model.minimize(obj_expr)
    _lambda_scaled = lambda_min + ((lambda_max - lambda_min) * _lambda)

    # Infinity norm constraints: -lambda <= (S[row,:] dot w) - e_i[row] <= lambda
    for row in range(p):
        lhs = model.sum(S[row, col] * w_vars[col] for col in range(p)) - e_i[row]
        model.add_constraint(lhs <= _lambda_scaled)
        model.add_constraint(lhs >= -_lambda_scaled)

    if fixed_values is not None:
        for j, val in fixed_values.items():
            model.add_constraint(w_vars[j] == val)

    solution = model.solve()
    if solution is None:
        model.clear()
        raise RuntimeError("No feasible solution found for column.")

    beta = np.array([solution.get_value(wv) for wv in w_vars], dtype=float)
    model.clear()
    return beta


def cde_v1(S, _lambda):
    p = S.shape[0]
    Omega_hat = np.zeros((p, p), dtype=float)
    for i in range(p):
        e_i = np.zeros(p)
        e_i[i] = 1.0

        fixed_dict = {}
        for j in range(i):
            fixed_dict[j] = Omega_hat[i, j]

        lambda_max, scaling_factor = find_lambda_max_cde_v1(S, e_i, p, 1, fixed_dict)
        lambda_min = lambda_max * find_lambda_min_cde_v1(
            S, e_i, scaling_factor, fixed_dict
        )
        if lambda_max <= 0:
            raise Exception("Lambda_max=0!")

        beta = cde_column_v1(S, e_i, _lambda, lambda_max, lambda_min, fixed_dict)

        Omega_hat[:, i] = beta

    return Omega_hat


def clime(S, _lambda):
    p = S.shape[0]
    Omega_hat = np.zeros((p, p), dtype=float)
    for i in range(p):
        e_i = np.zeros(p)
        e_i[i] = 1.0

        # lambda_max, scaling_factor = find_lambda_max_cde_v1(S, e_i, p, 1)
        # lambda_min = lambda_max * find_lambda_min_cde_v1(S, e_i, scaling_factor)
        lambda_min = find_lambda_min_cde_v1(S, e_i, 1)

        # if lambda_max <= 0:
        #     raise Exception("Lambda_max=0!")

        beta = cde_column_v1(S, e_i, _lambda, 1, lambda_min)

        Omega_hat[:, i] = beta

    for i in range(p):
        for j in range(i + 1, p):
            if np.abs(Omega_hat[i, j]) <= np.abs(Omega_hat[j, i]):
                Omega_hat[j, i] = Omega_hat[i, j]
            else:
                Omega_hat[i, j] = Omega_hat[j, i]

    return Omega_hat


def find_lambda_min_cde_v2(S):
    lambda_list = [i / 20 for i in range(1, 20)]
    left = 0
    right = len(lambda_list) - 1
    feasible_lambda = None

    while left <= right:
        mid = (left + right) // 2
        lam = lambda_list[mid]
        try:
            _ = cde_v2(S, lam)
            feasible_lambda = lam
            right = mid - 1
        except RuntimeError:
            left = mid + 1
    return feasible_lambda if feasible_lambda is not None else 1


def cde_v2(S, _lambda):
    p = S.shape[0]
    # start_time = time.time()
    model = Model(name="CDE")

    W = {
        (i, j): model.continuous_var(lb=-model.infinity, name=f"w_{i}_{j}")
        for i in range(p)
        for j in range(p)
    }

    obj_expr = model.sum(model.abs(W[(i, j)]) for i in range(p) for j in range(p))
    model.minimize(obj_expr)

    for i in range(p):
        for r in range(p):
            # Compute the dot product: (Sigma[r,:] dot W[:,i])
            lhs = model.sum(S[r, j] * W[(j, i)] for j in range(p))
            # Subtract the i-th column of identity: 1 if r == i, else 0.
            target = 1.0 if r == i else 0.0
            lhs = lhs - target
            model.add_constraint(lhs <= _lambda)
            model.add_constraint(lhs >= -_lambda)

    for i in range(p):
        for j in range(i + 1, p):
            model.add_constraint(W[(i, j)] == W[(j, i)])

    solution = model.solve()
    # print("--- %s seconds ---" % (time.time() - start_time))
    if solution is None:
        model.clear()
        raise RuntimeError("No feasible solution found for column.")

    W_est = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            W_est[i, j] = solution.get_value(W[(i, j)])

    return W_est


def find_lambda_max_cde_v3(S, p, factor):
    model = Model("lp1")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    W = {
        (i, j): model.continuous_var(lb=-model.infinity, name=f"w_{i}_{j}")
        for i in range(p)
        for j in range(p)
    }

    obj_expr = model.sum(model.abs(W[(i, j)]) for i in range(p) for j in range(p))
    model.minimize(obj_expr)

    for i in range(p):
        for j in range(i + 1, p):
            model.add_constraint(W[(i, j)] == W[(j, i)])

    for i in range(p):
        model.add_constraint(W[(i, i)] == 1)

    solution = model.solve()
    if solution is None or model.solve_status.value != 2:
        model.clear()
        raise Exception(
            "Infeasible lp1: Could not find a feasible solution to the baseline problem."
        )

    lp1_norm = solution.get_objective_value()
    model.clear()

    model = Model("lp2")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    W = {
        (i, j): model.continuous_var(lb=-model.infinity, name=f"w_{i}_{j}")
        for i in range(p)
        for j in range(p)
    }
    lambda_var = model.continuous_var(name="lambda")

    model.minimize(lambda_var)

    for i in range(p):
        for r in range(p):
            lhs = model.sum(S[r, j] * W[(j, i)] for j in range(p))
            target = 1.0 if r == i else 0.0
            lhs = lhs - target
            model.add_constraint(lhs <= lambda_var)
            model.add_constraint(lhs >= -lambda_var)

    for i in range(p):
        for j in range(i + 1, p):
            model.add_constraint(W[(i, j)] == W[(j, i)])

    for i in range(p):
        model.add_constraint(W[(i, i)] == 1)

    # Constraint: sum_{i,j} abs(W[i,j]) must equal the lp1_norm from LP1.
    expr = model.sum(model.abs(W[(i, j)]) for i in range(p) for j in range(p))
    model.add_constraint(expr == lp1_norm)

    solution = model.solve()
    if solution is None or model.solve_status.value != 2:
        model.clear()
        raise Exception("Infeasible lp2: Could not find a feasible lambda.")

    lambda_max = solution.get_value(lambda_var)
    new_factor = (1 / lambda_max * factor) if lambda_max != 0 else factor
    model.clear()
    return lambda_max


def find_lambda_min_cde_v3(S):
    lambda_list = [i / 20 for i in range(1, 20)]
    left = 0
    right = len(lambda_list) - 1
    feasible_lambda = None

    while left <= right:
        mid = (left + right) // 2
        lam = lambda_list[mid]
        try:
            _ = cde_v3(S, lam)
            feasible_lambda = lam
            right = mid - 1
        except RuntimeError:
            left = mid + 1
    return feasible_lambda if feasible_lambda is not None else 1


def cde_v3(S, _lambda):
    p = S.shape[0]
    # start_time = time.time()
    model = Model(name="CDE")

    W = {
        (i, j): model.continuous_var(lb=-model.infinity, name=f"w_{i}_{j}")
        for i in range(p)
        for j in range(p)
    }

    obj_expr = model.sum(model.abs(W[(i, j)]) for i in range(p) for j in range(p))
    model.minimize(obj_expr)

    for i in range(p):
        for r in range(p):
            # Compute the dot product: (Sigma[r,:] dot W[:,i])
            lhs = model.sum(S[r, j] * W[(j, i)] for j in range(p))
            # Subtract the i-th column of identity: 1 if r == i, else 0.
            target = 1.0 if r == i else 0.0
            lhs = lhs - target
            model.add_constraint(lhs <= _lambda)
            model.add_constraint(lhs >= -_lambda)

    for i in range(p):
        for j in range(i + 1, p):
            model.add_constraint(W[(i, j)] == W[(j, i)])

    for i in range(p):
        model.add_constraint(W[(i, i)] == 1)

    solution = model.solve()
    # print("--- %s seconds ---" % (time.time() - start_time))
    if solution is None:
        model.clear()
        raise RuntimeError("No feasible solution found for column.")

    W_est = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            W_est[i, j] = solution.get_value(W[(i, j)])

    return W_est


def generate_omega_model1(p):
    Omega = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            Omega[i, j] = 0.6 ** abs(i - j)
    return Omega


def generate_omega_model2(p):
    B = np.zeros((p, p))
    for i in range(p):
        for j in range(i + 1, p):
            B[i, j] = 0.5 if np.random.rand() < 0.1 else 0
            B[j, i] = B[i, j]

    eigenvalues = np.linalg.eigvalsh(B)
    lambda_max, lambda_min = max(eigenvalues), min(eigenvalues)
    delta = (lambda_max - p * lambda_min) / (p - 1)
    Omega = B + delta * np.eye(p)

    v = 1 / np.sqrt(np.diag(Omega))
    standardized_Omega = Omega * np.outer(v, v)

    return standardized_Omega


def generate_synthetic_data(Omega, n):
    p = Omega.shape[0]
    Sigma = np.linalg.inv(Omega)
    X = multivariate_normal.rvs(mean=np.zeros(p), cov=Sigma, size=n)
    return X


def frobenius_norm_error(Omega_true, Omega_est):
    return np.linalg.norm(Omega_true - Omega_est, "fro")


def l1_norm_error(Omega_true, Omega_est):
    return np.sum(np.abs(Omega_true - Omega_est))


def relative_frobenius_norm_error(Omega_true, Omega_est):
    return np.linalg.norm(Omega_true - Omega_est, "fro") / np.linalg.norm(
        Omega_true, "fro"
    )


def relative_l1_norm_error(Omega_true, Omega_est):
    return np.sum(np.abs(Omega_true - Omega_est)) / np.sum(np.abs(Omega_true))


def compute_tpr_fpr(true_Omega, est_Omega):
    true_support = (np.abs(true_Omega) > 1e-12).astype(int)
    est_support = (np.abs(est_Omega) > 1e-12).astype(int)

    TP = np.sum((true_support == 1) & (est_support == 1))
    FP = np.sum((true_support == 0) & (est_support == 1))
    TN = np.sum((true_support == 0) & (est_support == 0))
    FN = np.sum((true_support == 1) & (est_support == 0))

    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0

    return tpr, fpr


if __name__ == "__main__":
    try:
        results = {}
        lambda_list = [i / 20 for i in range(9)]

        for p, n in [
            (20, 50),
            (20, 35),
            (20, 20),
            (20, 10),
        ]:
            condition_key = f"p_{p}_n_{n}"
            results[condition_key] = {}

            I = np.eye(p)
            for model_name, generate_omega in [
                ("Model 1", generate_omega_model1),
                ("Model 2", generate_omega_model2),
            ]:
                print(f"p = {p}, {model_name}")
                stored_Omega = {}
                # set seed for reproducibility as n and create one true omega for all 100 simulations
                np.random.seed(n)
                true_Omega = generate_omega(p)
                for count in range(100):
                    print(f"count = {count}")
                    np.random.seed(count + 1)
                    start_time = time.time()
                    synthetic_data = generate_synthetic_data(true_Omega, n)
                    S = np.cov(synthetic_data, rowvar=False)
                    lambda_errors = {}
                    for _lambda in lambda_list:
                        cde_v1_err, cde_v2_err, cde_v3_err, clime_err = [], [], [], []
                        kf = KFold(n_splits=5, shuffle=True, random_state=count + 1)
                        fold = 1
                        for train_index, test_index in kf.split(synthetic_data):
                            # print(f"Fold = {fold}")
                            train_data = synthetic_data[train_index]
                            test_data = synthetic_data[test_index]

                            S_train = np.cov(train_data, rowvar=False)
                            S_test = np.cov(test_data, rowvar=False)

                            est_Omega_cde_v1 = cde_v1(S_train, _lambda)
                            cde_v1_err.append(
                                np.trace(est_Omega_cde_v1 @ S_test)
                                - np.log(np.linalg.det(est_Omega_cde_v1))
                            ) if np.linalg.det(
                                est_Omega_cde_v1
                            ) > 0 else cde_v1_err.append(1e99)

                            lambda_min_cde_v2 = find_lambda_min_cde_v2(S_train)
                            _lambda_scaled_cde_v2 = lambda_min_cde_v2 + (
                                (1 - lambda_min_cde_v2) * _lambda
                            )
                            est_Omega_cde_v2 = cde_v2(S_train, _lambda_scaled_cde_v2)
                            cde_v2_err.append(
                                np.trace(est_Omega_cde_v2 @ S_test)
                                - np.log(np.linalg.det(est_Omega_cde_v2))
                            ) if np.linalg.det(
                                est_Omega_cde_v2
                            ) > 0 else cde_v2_err.append(1e99)

                            lambda_min_cde_v3 = find_lambda_min_cde_v3(S_train)
                            lambda_max_cde_v3 = find_lambda_max_cde_v3(S_train, p, 1)
                            _lambda_scaled_cde_v3 = lambda_min_cde_v3 + (
                                (lambda_max_cde_v3 - lambda_min_cde_v3) * _lambda
                            )
                            est_Omega_cde_v3 = cde_v3(S_train, _lambda_scaled_cde_v3)
                            cde_v3_err.append(
                                np.trace(est_Omega_cde_v3 @ S_test)
                                - np.log(np.linalg.det(est_Omega_cde_v3))
                            ) if np.linalg.det(
                                est_Omega_cde_v3
                            ) > 0 else cde_v3_err.append(1e99)

                            est_Omega_clime = clime(S_train, _lambda)
                            clime_err.append(
                                np.trace(est_Omega_clime @ S_test)
                                - np.log(np.linalg.det(est_Omega_clime))
                            ) if np.linalg.det(
                                est_Omega_clime
                            ) > 0 else clime_err.append(1e99)

                            fold += 1

                        else:
                            cde_v1_err_final = sum(cde_v1_err) / len(cde_v1_err)
                            cde_v2_err_final = sum(cde_v2_err) / len(cde_v2_err)
                            cde_v3_err_final = sum(cde_v3_err) / len(cde_v3_err)
                            clime_err_final = sum(clime_err) / len(clime_err)
                            if _lambda not in lambda_errors:
                                lambda_errors[_lambda] = {}
                            lambda_errors[_lambda]["cde_v1"] = cde_v1_err_final
                            lambda_errors[_lambda]["cde_v2"] = cde_v2_err_final
                            lambda_errors[_lambda]["cde_v3"] = cde_v3_err_final
                            lambda_errors[_lambda]["clime"] = clime_err_final

                    else:
                        (
                            best_lambda_cde_v1,
                            best_lambda_cde_v2,
                            best_lambda_cde_v3,
                            best_lambda_clime,
                        ) = (
                            None,
                            None,
                            None,
                            None,
                        )
                        error_cde_v1, error_cde_v2, error_cde_v3, error_clime = (
                            None,
                            None,
                            None,
                            None,
                        )
                        for i in lambda_errors.keys():
                            if (error_cde_v1 is None) or (
                                lambda_errors[i]["cde_v1"] < error_cde_v1
                            ):
                                error_cde_v1 = lambda_errors[i]["cde_v1"]
                                best_lambda_cde_v1 = i

                            if (error_cde_v2 is None) or (
                                lambda_errors[i]["cde_v2"] < error_cde_v2
                            ):
                                error_cde_v2 = lambda_errors[i]["cde_v2"]
                                best_lambda_cde_v2 = i

                            if (error_cde_v3 is None) or (
                                lambda_errors[i]["cde_v3"] < error_cde_v3
                            ):
                                error_cde_v3 = lambda_errors[i]["cde_v3"]
                                best_lambda_cde_v3 = i

                            if (error_clime is None) or (
                                lambda_errors[i]["clime"] < error_clime
                            ):
                                error_clime = lambda_errors[i]["clime"]
                                best_lambda_clime = i

                    est_Omega_cde_v1 = cde_v1(S, best_lambda_cde_v1)

                    lambda_min_cde_v2 = find_lambda_min_cde_v2(S)
                    _lambda_scaled_cde_v2 = lambda_min_cde_v2 + (
                        (1 - lambda_min_cde_v2) * best_lambda_cde_v2
                    )
                    est_Omega_cde_v2 = cde_v2(S, _lambda_scaled_cde_v2)

                    lambda_min_cde_v3 = find_lambda_min_cde_v3(S)
                    lambda_max_cde_v3 = find_lambda_max_cde_v3(S, p, 1)
                    _lambda_scaled_cde_v3 = lambda_min_cde_v3 + (
                        (lambda_max_cde_v3 - lambda_min_cde_v3) * best_lambda_cde_v3
                    )
                    est_Omega_cde_v3 = cde_v3(S, _lambda_scaled_cde_v3)

                    est_Omega_clime = clime(S, best_lambda_clime)

                    stored_Omega[count] = {
                        "cde_v1": est_Omega_cde_v1,
                        "cde_v2": est_Omega_cde_v2,
                        "cde_v3": est_Omega_cde_v3,
                        "clime": est_Omega_clime,
                        "true": true_Omega,
                    }
                    print("--- %s seconds ---" % (time.time() - start_time))
                    np.savez(f"omega_{model_name}_{p}.npz", results=stored_Omega)

                results[condition_key][model_name] = stored_Omega
            np.savez(f"final_results_{model_name}_p_{p}_n_{n}.npz", results=results)

    except Exception as e:
        print(e)
        raise RuntimeError
    else:
        pass

    data = []

    for condition_key, models in results.items():
        # Extract p and n from the condition key, e.g., "p_20_n_40"
        parts = condition_key.split("_")
        p_val = int(parts[1])
        n_val = int(parts[3])

        # For each model in the condition:
        for model_name, sim_dict in models.items():
            # Define which methods we have.
            methods = ["cde_v1", "cde_v2", "cde_v3", "clime"]
            # Create a container to collect metrics across simulation counts for each method.
            aggregated = {
                method: {
                    "positive_definite": [],
                    "frobenius_error": [],
                    "l1_error": [],
                    "relative_frobenius_error": [],
                    "relative_l1_error": [],
                    "TPR": [],
                    "FPR": [],
                }
                for method in methods
            }

            # Loop over simulation counts.
            for count, simulation in sim_dict.items():
                true_Omega = simulation["true"]
                for method in methods:
                    est_Omega = simulation[method]
                    # Check for positive definiteness (convert boolean to 1/0).
                    pd_test = np.sum(np.abs(np.linalg.eigvals(est_Omega)) <= 0) == 0
                    # Compute TPR and FPR.
                    tpr, fpr = compute_tpr_fpr(true_Omega, est_Omega)
                    # Append the metrics for this simulation.
                    aggregated[method]["positive_definite"].append(1 if pd_test else 0)
                    aggregated[method]["frobenius_error"].append(
                        frobenius_norm_error(true_Omega, est_Omega)
                    )
                    aggregated[method]["l1_error"].append(
                        l1_norm_error(true_Omega, est_Omega)
                    )
                    aggregated[method]["relative_frobenius_error"].append(
                        relative_frobenius_norm_error(true_Omega, est_Omega)
                    )
                    aggregated[method]["relative_l1_error"].append(
                        relative_l1_norm_error(true_Omega, est_Omega)
                    )
                    aggregated[method]["TPR"].append(tpr)
                    aggregated[method]["FPR"].append(fpr)

            # Now average the metrics across counts for each method.
            for method in methods:
                avg_positive_definite = np.mean(aggregated[method]["positive_definite"])
                avg_frobenius_error = np.mean(aggregated[method]["frobenius_error"])
                avg_l1_error = np.mean(aggregated[method]["l1_error"])
                avg_relative_frobenius_error = np.mean(
                    aggregated[method]["relative_frobenius_error"]
                )
                avg_relative_l1_error = np.mean(aggregated[method]["relative_l1_error"])
                avg_TPR = np.mean(aggregated[method]["TPR"])
                avg_FPR = np.mean(aggregated[method]["FPR"])

                # Append a row for this combination.
                data.append(
                    {
                        "model": model_name,
                        "method": method,
                        "p": p_val,
                        "n": n_val,
                        "positive_definite": avg_positive_definite,
                        "frobenius_error": avg_frobenius_error,
                        "l1_error": avg_l1_error,
                        "relative_frobenius_error": avg_relative_frobenius_error,
                        "relative_l1_error": avg_relative_l1_error,
                        "TPR": avg_TPR,
                        "FPR": avg_FPR,
                    }
                )

    df_results = pd.DataFrame(data)

    print("Done!")


def load_results(path):
    data = np.load(path, allow_pickle=True)
    if "results" not in data:
        raise ValueError("Expected an array named 'results' in the npz.")
    raw = data["results"]

    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.shape == ():
        raw = raw.item()

    cleaned = {}
    for run, val in raw.items():
        if isinstance(val, np.ndarray) and val.dtype == object and val.shape == ():
            val = val.item()
        cleaned[run] = val

    sample = next(iter(cleaned.values()))
    if isinstance(sample, dict) and all(isinstance(k, int) for k in sample):
        peeled = {}
        for run_outer, inner in cleaned.items():
            inner_dict = next(iter(inner.values()))
            peeled[run_outer] = inner_dict
        cleaned = peeled

    return cleaned

def zero_frequency_matrix(results, method, threshold=1e-6):
    runs = sorted(results.keys())
    p = results[runs[0]][method].shape[0]
    freq = np.zeros((p, p), float)

    for run in runs:
        M = results[run][method]
        freq += (np.abs(M) < threshold).astype(float)

    return freq / len(runs)

def create_model_figure(results, method_labels, threshold=1e-2):
    methods = ["true", "cde_v1", "cde_v2", "cde_v3", "clime"]
    fig = make_subplots(
        rows=1, cols=len(methods),
        horizontal_spacing=0.02,
    )

    for col, (m, label) in enumerate(zip(methods, method_labels), start=1):
        Z = zero_frequency_matrix(results, m, threshold=threshold)
        fig.add_trace(
            go.Heatmap(z=Z, zmin=0, zmax=1,
                       colorscale=[[0, "black"], [1, "white"]],
                       showscale=False),
            row=1, col=col
        )
        xref = f"x{col if col > 1 else ''} domain"
        yref = f"y{col if col > 1 else ''} domain"

        titles = {
            'cde_v3': 'CDE V3',
            'cde_v2': 'CDE V2',
            'cde_v1': 'CDE V1',
            'clime': 'CLIME',
            'true': 'True',
        }
        fig.add_annotation(
            xref=xref, yref=yref,
            x=0.5,
            y=-0.10,
            text=titles[label],
            showarrow=False,
            font=dict(size=14)
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        margin=dict(t=50, b=50),
        height=425, width=1800
    )
    return fig


# This part is to load the data from n=10 p=20 for both models and plot frequency of 0s and 1s
# if __name__ == "__main__":
#     results1 = load_results("final_results_Model 1_p_20_n_10.npz")
#     results1 = results1["p_20_n_10"]["Model 1"]
#
#     results2 = load_results("final_results_Model 2_p_20_n_10.npz")
#     results2 = results2["p_20_n_10"]["Model 2"]
#
#     method_labels = ["true", "cde_v1", "cde_v2", "cde_v3", "clime"]
#
#     fig1 = create_model_figure(results1, method_labels, threshold=1e-2)
#     fig1.show()
#
#     fig2 = create_model_figure(results2, method_labels, threshold=1e-6)
#     fig2.show()

