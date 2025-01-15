import numpy as np
from scipy.stats import multivariate_normal
from docplex.mp.model import Model
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import requests
from numpy.linalg import eigvals

np.random.seed(123)


def find_lambda_max_column(S, e_i, p):
    # TODO: No phase 1 since no linear equality constraints so no benchmark beta
    # TODO: Minimising lambda subject to constraint involving beta which is unknown

    model = Model("lambda_max_column")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    w_vars = model.continuous_var_list(p, lb=-model.infinity, name="beta")
    _lambda_scaled = model.continuous_var(name="lambda")

    model.minimize(_lambda_scaled)

    # Infinity norm constraints: -lambda <= (S[row,:] dot w) - e_i[row] <= lambda
    for row in range(p):
        lhs = model.sum(S[row, col] * w_vars[col] for col in range(p)) - e_i[row]
        model.add_constraint(lhs <= _lambda_scaled)
        model.add_constraint(lhs >= -_lambda_scaled)

    solution = model.solve()
    if solution is None:
        raise RuntimeError("No feasible solution found for lambda_max.")

    lambda_max = solution.get_value(_lambda_scaled)
    scaling_factor = 1 / lambda_max if lambda_max != 0 else 1.0
    model.clear()

    return lambda_max, scaling_factor


def cde_column(S, e_i, _lambda_scaled, scaling_factor, fixed_values=None):
    p = S.shape[0]
    model = Model(name="CDE_col")
    model.context.solver.log_output = False  # Hide solver logs

    w_vars = model.continuous_var_list(p, lb=-model.infinity, name="beta")

    obj_expr = model.sum(model.abs(w_vars[j]) for j in range(p))
    model.minimize(obj_expr)

    # Infinity norm constraints: -lambda <= (S[row,:] dot w) - e_i[row] <= lambda
    for row in range(p):
        if np.all(S[row] == 0):
            continue  # Skip rows with all zeros to avoid infeasibility
        lhs = model.sum(S[row, col] * w_vars[col] for col in range(p)) - e_i[row]
        model.add_constraint(lhs <= _lambda_scaled / scaling_factor)
        model.add_constraint(lhs >= -_lambda_scaled / scaling_factor)

    if fixed_values is not None:
        for j, val in fixed_values.items():
            model.add_constraint(w_vars[j] == val)

    solution = model.solve()
    if solution is None:
        raise RuntimeError("No feasible solution found for column.")

    beta = np.array([solution.get_value(wv) for wv in w_vars], dtype=float)
    model.clear()
    return beta


def cde_columnwise_recursive(S, lambda_list):
    # TODO: For some reason greater lambda results in more sparsity
    p = S.shape[0]
    Theta_hat = np.zeros((p, p), dtype=float)
    lambdas_used = {}
    for i in range(p):
        print(f"Processing Column {i + 1}/{p}")
        e_i = np.zeros(p)
        e_i[i] = 1.0

        fixed_dict = {}
        for j in range(i):
            fixed_dict[j] = Theta_hat[i, j]

        lambda_max, scaling_factor = find_lambda_max_column(S, e_i, p)
        if lambda_max <= 0:
            pass
            # raise Exception("Lambda_max=0!")

        lam = None
        col_solved = False
        for _lambda in lambda_list:
            try:
                beta_i = cde_column(S, e_i, _lambda, scaling_factor, fixed_dict)
                if np.all(beta_i == 0):
                    print(f"All zeros for column {i + 1}, lambda = {_lambda}")
                    # raise SyntaxError
                col_solved = True
                lam = _lambda
                # print(f"worked for lambda = {lam}")
                break
            except RuntimeError:
                continue
            except SyntaxError:
                print("caught in zero loop")
                raise RuntimeError
        if col_solved:
            lambdas_used[i + 1] = lam
            if lam != lambda_list[0]:
                print(f"Used different lambda = {lam}")
            Theta_hat[:, i] = beta_i
        else:
            print(f"Column {i + 1} has no solution.")
            raise RuntimeError

    return Theta_hat, lambdas_used


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


def relative_spectral_norm_error(Theta_true, Theta_est):
    return np.linalg.norm(Theta_true - Theta_est, 2) / np.linalg.norm(Theta_true, 2)


def compute_support_metrics(Theta_true, Theta_est):
    true_support = (np.abs(Theta_true) > 1e-12).astype(int)
    est_support = (np.abs(Theta_est) > 1e-12).astype(int)

    true_positives = np.sum((true_support == 1) & (est_support == 1))
    false_positives = np.sum((true_support == 0) & (est_support == 1))
    false_negatives = np.sum((true_support == 1) & (est_support == 0))
    true_negatives = np.sum((true_support == 0) & (est_support == 0))

    tpr = true_positives / (true_positives + false_negatives + 1e-8)
    fpr = false_positives / (false_positives + true_negatives + 1e-8)

    hamming_distance = np.sum(true_support != est_support)

    return tpr, fpr, hamming_distance


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
    # TODO: Due to the way symmetry is being imposed, the IC for lambda is very sensitive (chaos theory???)
    # TODO: Must be symmetric, PD, S * Theta = I. Symmetry is enforced directly in the problem
    # TODO: In theory, Theta_est is assympotically PD according to Cai et al (Jianqing Fan et al)

    # Using GPT's method of creating data for simulation study
    # Generate Theta, derive S, generate synthetic data from multivariate normal
    p = 250
    n = 125
    sparsity = 0.98

    true_Theta = create_sparse_precision_matrix(p, sparsity, epsilon=1e-4)
    synthetic_data = generate_synthetic_data(true_Theta, n)
    S = np.cov(synthetic_data, rowvar=False)
    lambda_list = np.linspace(2.5, 5, 20)

    try:
        est_Theta = np.load("synt_results_250_125_098.npy")

        # est_Theta, lambdas = cde_columnwise_recursive(S, lambda_list)
        # # np.save("synt_results_250_125_098.npy", est_Theta)
        pass
    except Exception as e:
        print(e)
        response = requests.post(
            f"https://ntfy.sh/firaz_python",
            data="❌ Script Failed! Please check for errors.".encode("utf-8"),
            headers={
                "Title": "Script Failed".encode("utf-8").decode("latin-1"),
                "Priority": "high",
                "Tags": "cross,fire",
                "Sound": "siren",
            },
        )
    else:
        response = requests.post(
            f"https://ntfy.sh/firaz_python",
            data="✅ Script finished running".encode("utf-8"),
            headers={
                "Title": "Script Completed".encode("utf-8").decode("latin-1"),
                "Priority": "high",
                "Tags": "check",
            },
        )

    # Check est_Theta for positive definiteness
    eigenvalues = eigvals(est_Theta)
    if np.sum(np.abs(eigenvalues) <= 0) == 0:
        print("Passed Positive Definite test")
    else:
        print("Failed Positive Definite test")

    # Metrics for checking accuracy of estimated theta against True data
    frobenius_error = frobenius_norm_error(true_Theta, est_Theta)
    l1_error = l1_norm_error(true_Theta, est_Theta)
    rel_spectral_error = relative_spectral_norm_error(true_Theta, est_Theta)
    tpr, fpr, hamming_dist = compute_support_metrics(true_Theta, est_Theta)
    cond_number = np.linalg.cond(est_Theta)

    print(f"Frobenius norm error: {frobenius_error:.4f}")
    print(f"L1-Norm Error: {l1_error:.4f}")
    print(f"Relative Spectral Norm Error: {rel_spectral_error:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Hamming Distance: {hamming_dist}")
    print(f"Condition Number: {cond_number:.4f}")

    fig_sparsity, fig_magnitudes = plot_sparsity_and_magnitude(
        true_Theta, est_Theta, p, n
    )
    # fig_sparsity.write_html("sparsity_plot.html")
    # fig_magnitudes.write_html("magnitude_plot.html")
    print("Done!")
