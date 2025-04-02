import numpy as np
from scipy.stats import multivariate_normal
from docplex.mp.model import Model
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from numpy.linalg import eigvals
import pandas as pd
import requests
import time

np.random.seed(123)


def find_lambda_max_column(S, e_i, p, factor, fixed_values=None):
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


def find_lambda_min_column(S, e_i, scaling_factor, fixed_values=None):
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
    return feasible_lambda


def cde_column(S, e_i, _lambda_scaled, scaling_factor, fixed_values=None):
    p = S.shape[0]
    model = Model(name="CDE_col")
    model.context.solver.log_output = False  # Hide solver logs

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


def cde_column_v2(S, e_i, _lambda, lambda_max, lambda_min, fixed_values=None):
    p = S.shape[0]
    model = Model(name="CDE_col")
    model.context.solver.log_output = False  # Hide solver logs

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


def cde_columnwise_recursive(S, lambda_list):
    p = S.shape[0]
    Omega_hat = np.zeros((p, p), dtype=float)
    for i in range(p):
        print(f"CDE Processing Column {i + 1}/{p}")
        e_i = np.zeros(p)
        e_i[i] = 1.0

        fixed_dict = {}
        for j in range(i):
            fixed_dict[j] = Omega_hat[i, j]

        lambda_max, scaling_factor = find_lambda_max_column(S, e_i, p, 1, fixed_dict)
        lambda_min = lambda_max * find_lambda_min_column(
            S, e_i, scaling_factor, fixed_dict
        )
        if lambda_max <= 0:
            raise Exception("Lambda_max=0!")

        if i == 0:
            best_norm = 1e99
            for _lambda in lambda_list:
                beta_i = cde_column_v2(
                    S, e_i, _lambda, lambda_max, lambda_min, fixed_dict
                )
                # pick best performing l2 norm
                norm = np.linalg.norm(beta_i - true_Omega[:, i], 2) / np.linalg.norm(
                    true_Omega[:, i], 2
                )
                if norm < best_norm:
                    best_norm = norm
                    lam = _lambda
                    best_beta = beta_i
            else:
                print(f"lambda used: {lam}")
        else:
            best_beta = cde_column_v2(S, e_i, lam, lambda_max, lambda_min, fixed_dict)
        Omega_hat[:, i] = best_beta

    return Omega_hat


def clime(S, lambda_list):
    p = S.shape[0]
    Omega_hat = np.zeros((p, p), dtype=float)
    for i in range(p):
        print(f"CLIME Processing Column {i + 1}/{p}")
        e_i = np.zeros(p)
        e_i[i] = 1.0

        lambda_max, scaling_factor = find_lambda_max_column(S, e_i, p, 1)
        lambda_min = lambda_max * find_lambda_min_column(S, e_i, scaling_factor)

        if lambda_max <= 0:
            raise Exception("Lambda_max=0!")

        if i == 0:
            best_norm = 1e99
            for _lambda in lambda_list:
                beta_i = cde_column_v2(S, e_i, _lambda, lambda_max, lambda_min)
                # pick best performing relative l2 norm
                norm = np.linalg.norm(beta_i - true_Omega[:, i], 2) / np.linalg.norm(
                    true_Omega[:, i], 2
                )
                if norm < best_norm:
                    best_norm = norm
                    lam = _lambda
                    best_beta = beta_i
            else:
                print(f"lambda used: {lam}")
        else:
            best_beta = cde_column_v2(S, e_i, lam, lambda_max, lambda_min)

        Omega_hat[:, i] = best_beta

    for i in range(p):
        for j in range(i + 1, p):
            if np.abs(Omega_hat[i, j]) <= np.abs(Omega_hat[j, i]):
                Omega_hat[j, i] = Omega_hat[i, j]
            else:
                Omega_hat[i, j] = Omega_hat[j, i]

    # Omega_hat = 0.5 * (Omega_hat + Omega_hat.T)

    return Omega_hat


def cde_new(S, lambda_scaled):
    p = S.shape[0]
    start_time = time.time()
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
            model.add_constraint(lhs <= lambda_scaled)
            model.add_constraint(lhs >= -lambda_scaled)

    for i in range(p):
        for j in range(i + 1, p):
            model.add_constraint(W[(i, j)] == W[(j, i)])

    solution = model.solve()
    print("--- %s seconds ---" % (time.time() - start_time))
    if solution is None:
        model.clear()
        return "No feasible solution found for the precision matrix."

    W_est = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            W_est[i, j] = solution.get_value(W[(i, j)])

    return W_est


def find_lambda_max_new(S, p, factor):
    model = Model("lp1")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    W = {
        (i, j): model.continuous_var(lb=-model.infinity, name=f"w_{i}_{j}")
        for i in range(p)
        for j in range(p)
    }

    # Objective: minimize sum of absolute values of all entries of W
    obj_expr = model.sum(model.abs(W[(i, j)]) for i in range(p) for j in range(p))
    model.minimize(obj_expr)

    # Add symmetry constraints: W[i,j] = W[j,i] for all i<j.
    for i in range(p):
        for j in range(i + 1, p):
            model.add_constraint(W[(i, j)] == W[(j, i)])

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
    model.context.solver.log_output = False

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

    # Apply symmetry constraints
    for i in range(p):
        for j in range(i + 1, p):
            model.add_constraint(W[(i, j)] == W[(j, i)])

    expr = model.sum(model.abs(W[(i, j)]) for i in range(p) for j in range(p))
    model.add_constraint(expr == lp1_norm)

    solution = model.solve()
    if solution is None or model.solve_status.value != 2:
        model.clear()
        raise Exception("Infeasible lp2: Could not find a feasible lambda.")

    lambda_max = solution.get_value(lambda_var)
    new_factor = (1 / lambda_max * factor) if lambda_max != 0 else factor
    model.clear()
    return lambda_max, new_factor


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
        # results = {}
        #
        # for p, n in [(30, 100), (60, 100), (90, 100), (120, 100), (200, 100)]:
        #     lambda_list = [i / 20 for i in range(21)]
        #
        #     for model_name, generate_omega in [
        #         ("Model 1", generate_omega_model1),
        #         ("Model 2", generate_omega_model2),
        #     ]:
        #         print(f"p = {p}, {model_name}")
        #         true_Omega = generate_omega(p)
        #         synthetic_data = generate_synthetic_data(true_Omega, n)
        #
        #         S = np.cov(synthetic_data, rowvar=False)
        #         est_Omega_cde = cde_columnwise_recursive(S, lambda_list)
        #         est_Omega_clime = clime(S, lambda_list)
        #
        #         if p not in results:
        #             results[p] = {}
        #
        #         results[p][model_name] = {
        #             "cde": est_Omega_cde,
        #             "clime": est_Omega_clime,
        #             "true": true_Omega,
        #         }
        results = {}

        for p, n in [(20, 40), (40, 40), (60, 40), (80, 40)]:
            lambda_list_clime = [i / 50 for i in range(51)]
            lambda_list_cde = reversed([i / 50 for i in range(51)])

            for model_name, generate_omega in [
                ("Model 1", generate_omega_model1),
                ("Model 2", generate_omega_model2),
            ]:
                print(f"p = {p}, {model_name}")
                true_Omega = generate_omega(p)
                synthetic_data = generate_synthetic_data(true_Omega, n)
                S = np.cov(synthetic_data, rowvar=False)

                cde_error = 1e99
                for _lambda in lambda_list_cde:
                    est_Omega_cde1 = cde_new(S, _lambda)
                    if isinstance(est_Omega_cde1, np.ndarray):
                        e = relative_frobenius_norm_error(true_Omega, est_Omega_cde1)
                        if e < cde_error:
                            est_Omega_cde = est_Omega_cde1.copy()
                            cde_error = e
                            best_lambda = _lambda
                    else:
                        break
                print(f"best lambda CDE is:{best_lambda}")

                est_Omega_clime = clime(S, lambda_list_clime)

                if p not in results:
                    results[p] = {}

                results[p][model_name] = {
                    "cde": est_Omega_cde,
                    "clime": est_Omega_clime,
                    "true": true_Omega,
                }
        # save_dict = {
        #     f"{p}_{model_name}_{method}": Omega
        #     for p, models in results.items()
        #     for model_name, estimations in models.items()
        #     for method, Omega in estimations.items()
        # }
        #
        # np.savez("PrecisionResultsFinal.npz", **save_dict)
        #
        # loaded = np.load("PrecisionResultsFinal1.npz")
        #
        # loaded_results = {}
        # for key in loaded.keys():
        #     p, model_name, method = key.split("_", maxsplit=2)
        #     p = int(p)
        #
        #     if p not in loaded_results:
        #         loaded_results[p] = {}
        #     if model_name not in loaded_results[p]:
        #         loaded_results[p][model_name] = {}
        #
        #     loaded_results[p][model_name][method] = loaded[key]
        # results = loaded_results

    except Exception as e:
        print(e)
        # response = requests.post(
        #     f"https://ntfy.sh/firaz_python",
        #     data="❌ Script Failed! Please check for errors.".encode("utf-8"),
        #     headers={
        #         "Title": "Script Failed".encode("utf-8").decode("latin-1"),
        #         "Priority": "high",
        #         "Tags": "cross,fire",
        #         "Sound": "siren",
        #     },
        # )
        raise RuntimeError
    else:
        # response = requests.post(
        #     f"https://ntfy.sh/firaz_python",
        #     data="✅ Script finished running".encode("utf-8"),
        #     headers={
        #         "Title": "Script Completed".encode("utf-8").decode("latin-1"),
        #         "Priority": "high",
        #         "Tags": "check",
        #     },
        # )
        pass

    data = []

    for p, models in results.items():
        for model_name, estimations in models.items():
            est_Omega_cde, est_Omega_clime, true_Omega = (
                estimations["cde"],
                estimations["clime"],
                estimations["true"],
            )

            clime_pd_test = np.sum(np.abs(np.linalg.eigvals(est_Omega_clime)) <= 0) == 0
            cde_pd_test = np.sum(np.abs(np.linalg.eigvals(est_Omega_cde)) <= 0) == 0

            clime_tpr, clime_fpr = compute_tpr_fpr(true_Omega, est_Omega_clime)
            cde_tpr, cde_fpr = compute_tpr_fpr(true_Omega, est_Omega_cde)

            clime_metrics = {
                "model": model_name,
                "method": "CLIME",
                "size": p,
                "positive_definite": clime_pd_test,
                "frobenius_error": frobenius_norm_error(true_Omega, est_Omega_clime),
                "l1_error": l1_norm_error(true_Omega, est_Omega_clime),
                "relative_frobenius_error": relative_frobenius_norm_error(
                    true_Omega, est_Omega_clime
                ),
                "relative_l1_error": relative_l1_norm_error(
                    true_Omega, est_Omega_clime
                ),
                "TPR": clime_tpr,
                "FPR": clime_fpr,
            }

            cde_metrics = {
                "model": model_name,
                "method": "CDE",
                "size": p,
                "positive_definite": cde_pd_test,
                "frobenius_error": frobenius_norm_error(true_Omega, est_Omega_cde),
                "l1_error": l1_norm_error(true_Omega, est_Omega_cde),
                "relative_frobenius_error": relative_frobenius_norm_error(
                    true_Omega, est_Omega_cde
                ),
                "relative_l1_error": relative_l1_norm_error(true_Omega, est_Omega_cde),
                "TPR": cde_tpr,
                "FPR": cde_fpr,
            }

            data.append(clime_metrics)
            data.append(cde_metrics)

    df_results = pd.DataFrame(data)

    print("Done!")
