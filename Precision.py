import numpy as np
from scipy.stats import multivariate_normal
from docplex.mp.model import Model
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from numpy.linalg import eigvals
import pandas as pd

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
    # TODO: For some reason greater lambda results in more sparsity
    p = S.shape[0]
    Theta_hat = np.zeros((p, p), dtype=float)
    for i in range(p):
        print(f"CDE Processing Column {i + 1}/{p}")
        e_i = np.zeros(p)
        e_i[i] = 1.0

        fixed_dict = {}
        for j in range(i):
            fixed_dict[j] = Theta_hat[i, j]

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
                norm = np.linalg.norm(beta_i - true_Theta[:, i], 2) / np.linalg.norm(
                    true_Theta[:, i], 2
                )
                if norm < best_norm:
                    best_norm = norm
                    lam = _lambda
                    best_beta = beta_i
            else:
                print(f"lambda used: {lam}")
        else:
            # best_beta = cde_column(S, e_i, lam, scaling_factor, fixed_dict)
            best_beta = cde_column_v2(S, e_i, lam, lambda_max, lambda_min, fixed_dict)
        Theta_hat[:, i] = best_beta

    return Theta_hat


def clime(S, lambda_list):
    p = S.shape[0]
    Theta_hat = np.zeros((p, p), dtype=float)
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
                norm = np.linalg.norm(beta_i - true_Theta[:, i], 2) / np.linalg.norm(
                    true_Theta[:, i], 2
                )
                if norm < best_norm:
                    best_norm = norm
                    lam = _lambda
                    best_beta = beta_i
            else:
                print(f"lambda used: {lam}")
        else:
            best_beta = cde_column_v2(S, e_i, lam, lambda_max, lambda_min)

        Theta_hat[:, i] = best_beta

    Theta_hat = 0.5 * (Theta_hat + Theta_hat.T)
    return Theta_hat


def create_sparse_precision_matrix(p, sparsity=0.9, epsilon=1e-4):
    Theta = np.zeros((p, p))

    # Fill diagonal with large positive values to ensure positive definiteness
    for i in range(p):
        Theta[i, i] = (
            np.random.uniform(1, 2) + epsilon
        )  # Add small epsilon to ensure positivity

    # Introduce off-diagonal sparsity
    for i in range(p):
        for j in range(i + 1, p):
            if (
                np.random.rand() > sparsity
            ):  # Only add non-zero with probability (1 - sparsity)
                value = np.random.uniform(-0.5, 0.5)
                Theta[i, j] = value
                Theta[j, i] = value  # Ensure symmetry

    # Ensure the matrix is positive definite
    eigvals = np.linalg.eigvalsh(Theta)
    if np.any(eigvals <= 0):
        min_eigval = np.min(eigvals)
        Theta += np.eye(p) * (abs(min_eigval) + epsilon)

    return Theta


def generate_synthetic_data(Theta, n):
    p = Theta.shape[0]
    Sigma = np.linalg.inv(Theta + np.eye(p))  # Add small regularization to the inverse
    X = multivariate_normal.rvs(mean=np.zeros(p), cov=Sigma, size=n)
    return X


def frobenius_norm_error(Theta_true, Theta_est):
    return np.linalg.norm(Theta_true - Theta_est, "fro")


def l1_norm_error(Theta_true, Theta_est):
    return np.sum(np.abs(Theta_true - Theta_est))


def relative_frobenius_norm_error(Theta_true, Theta_est):
    return np.linalg.norm(Theta_true - Theta_est, "fro") / np.linalg.norm(
        Theta_true, "fro"
    )


def relative_l1_norm_error(Theta_true, Theta_est):
    return np.sum(np.abs(Theta_true - Theta_est)) / np.sum(np.abs(Theta_true))


def plot_sparsity_and_magnitude(Theta_true, Theta_est, p, n):
    true_support = (np.abs(Theta_true) > 1e-12).astype(int)
    est_support = (np.abs(Theta_est) > 1e-12).astype(int)

    fig_sparsity = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("True Sparsity Pattern", "Estimated Sparsity Pattern"),
    )

    fig_sparsity.add_trace(
        go.Heatmap(
            z=true_support[::-1],
            colorscale="Blues",
            showscale=False,
        ),
        row=1,
        col=1,
    )
    fig_sparsity.add_trace(
        go.Heatmap(
            z=est_support[::-1],
            colorscale="Blues",
            showscale=False,
        ),
        row=1,
        col=2,
    )

    fig_sparsity.update_layout(
        title_text=f"Sparsity Patterns of Precision Matrices (p = {p}, n = {n})",
    )
    fig_sparsity.update_xaxes(title_text="Index")
    fig_sparsity.update_yaxes(title_text="Index (reversed)")

    fig_magnitudes = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "True Precision Matrix (Magnitudes)",
            "Estimated Precision Matrix (Magnitudes)",
        ),
    )

    fig_magnitudes.add_trace(
        go.Heatmap(
            z=Theta_true[::-1],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Magnitude"),
        ),
        row=1,
        col=1,
    )
    fig_magnitudes.add_trace(
        go.Heatmap(
            z=Theta_est[::-1],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Magnitude"),
        ),
        row=1,
        col=2,
    )

    fig_magnitudes.update_layout(
        title_text=f"Magnitudes of Precision Matrices (p = {p}, n = {n})",
    )
    fig_magnitudes.update_xaxes(title_text="Index")
    fig_magnitudes.update_yaxes(title_text="Index (reversed)")

    fig_sparsity.show()
    fig_magnitudes.show()

    return fig_sparsity, fig_magnitudes


if __name__ == "__main__":
    # TODO: With reference to article (Jianqing Fan et al) using different lambda for each column CDE
    # TODO: Must be symmetric, PD, S * Theta = I. Symmetry is enforced directly in the problem
    # TODO: In theory, Theta_est is assympotically PD according to Cai et al (Jianqing Fan et al)
    # TODO: If true theta is very sparse, clime and cde no diff since they only have entries along diagonal

    try:
        results = {}

        for (p, n) in [(100, 127), (200, 253), (400, 505), (125, 100), (250,125)]:
            print(f"Sample size: {p}")
            sparsity = 0.90
            true_Theta = create_sparse_precision_matrix(p, sparsity, epsilon=1e-4)
            synthetic_data = generate_synthetic_data(true_Theta, n)

            S = np.cov(synthetic_data, rowvar=False)
            lambda_list = [i / 20 for i in range(21)]
            est_Theta_cde = cde_columnwise_recursive(S, lambda_list)
            est_Theta_clime = clime(S, lambda_list)
            results[p] = {"cde": est_Theta_cde, "clime": est_Theta_clime, "true": true_Theta}

        # save_dict = {
        #     f"{key}_{subkey}": value for key, subdict in results.items() for subkey, value in subdict.items()
        # }
        # np.savez("PrecisionResults.npz", **save_dict)

        # loaded = np.load("PrecisionResults.npz")
        # loaded = np.load("PrecisionResultsHD.npz")
        # for k in loaded.keys():
        #     key, subkey = k.split("_")
        #     key = int(key)  # Convert back to int
        #     if key not in results:
        #         results[key] = {}
        #     results[key][subkey] = loaded[k]

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

    for i in results:
        est_Theta_cde, est_Theta_clime, true_Theta = (
            results[i]["cde"],
            results[i]["clime"],
            results[i]["true"],
        )

        clime_pd_test = np.sum(np.abs(eigvals(est_Theta_clime)) <= 0) == 0
        cde_pd_test = np.sum(np.abs(eigvals(est_Theta_cde)) <= 0) == 0

        clime_metrics = {
            "method": "CLIME",
            "size": i,
            "positive_definite": clime_pd_test,
            "frobenius_error": frobenius_norm_error(true_Theta, est_Theta_clime),
            "l1_error": l1_norm_error(true_Theta, est_Theta_clime),
            "relative_frobenius_error": relative_frobenius_norm_error(
                true_Theta, est_Theta_clime
            ),
            "relative_l1_error": relative_l1_norm_error(true_Theta, est_Theta_clime),
        }

        cde_metrics = {
            "method": "CDE",
            "size": i,
            "positive_definite": cde_pd_test,
            "frobenius_error": frobenius_norm_error(true_Theta, est_Theta_cde),
            "l1_error": l1_norm_error(true_Theta, est_Theta_cde),
            "relative_frobenius_error": relative_frobenius_norm_error(
                true_Theta, est_Theta_cde
            ),
            "relative_l1_error": relative_l1_norm_error(true_Theta, est_Theta_cde),
        }

        data.append(clime_metrics)
        data.append(cde_metrics)

    df_results = pd.DataFrame(data)

    # fig_sparsity_cde, fig_magnitudes_cde = plot_sparsity_and_magnitude(
    #     true_Theta, est_Theta, p, n
    # )
    # fig_sparsity_clime, fig_magnitudes_clime = plot_sparsity_and_magnitude(
    #     true_Theta, est_Theta_clime, p, n
    # )

    # fig_sparsity.write_html("sparsity_plot.html")
    # fig_magnitudes.write_html("magnitude_plot.html")
    print("Done!")
