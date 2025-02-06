import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from docplex.mp.model import Model
import plotly.graph_objs as go
from plotly.subplots import make_subplots

np.random.seed(123)


def find_lambda_max_cplex(sigma, eta, A, b, p, k, factor=1.0):
    model = Model("lp1")
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))

    # Objective: minimize sum of abs(w[i])
    model.minimize(model.sum(model.abs(w[i]) for i in range(p)))

    # Add linear equality constraints A w = b
    for i in range(k):
        expr = model.dot(w, A[i])
        model.add_constraint(expr == b[i])

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

    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))
    _lambda_scaled = model.continuous_var(name='lambda')

    model.minimize(_lambda_scaled)

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

    for i in range(k):
        expr = model.dot(w, A[i])
        model.add_constraint(expr == b[i])

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


def CDE_DOcplex(sigma, eta, A, b, p, k, factor, _lambda_scaled):

    model = Model(name="CDE")

    # From Dechuan's reference code
    model.parameters.read.scale = -1
    model.parameters.lpmethod = 4

    w = np.array(model.continuous_var_list([f"w{i}" for i in range(p)], lb=-model.infinity))
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

    # Equality constraints A w = b
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

    model = Model("constrained_lasso")

    w = model.continuous_var_list(p, name="w", lb=-model.infinity)
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


def compute_reach_and_ctr(page_view_matrix, site_info, beta):

    pvm = page_view_matrix.copy()
    si = site_info.copy()

    # Step 1: Prepare the data and align indices
    si = si.set_index("Site_Name")  # Set index to Site_Name
    pvm = pvm.fillna(0)  # Replace NA values with 0

    # Step 2: Compute gamma (fraction of ads per dollar spent)
    cost_per_thousand = si["Cost"].values  # Cost per thousand impressions (CPM)
    tau = si["Pages"].values  # Total website visits (Pages)
    gamma = 1 / (cost_per_thousand * tau)  # Gamma calculation

    beta = np.array(beta)

    beta_gamma = beta * gamma  # Element-wise multiplication of beta and gamma

    # Compute the term for Reach: (1 - beta_j * gamma_j) raised to the power of z_ij
    adjusted_reach_matrix = np.power(
        1 - beta_gamma, pvm.values
    )  # (1 - beta_j * gamma_j) ** z_ij
    adjusted_reach_matrix = np.clip(adjusted_reach_matrix, 0, 1)  # keep in range [0, 1]

    # Step 5: Compute Reach based on PaC article
    # Product over all websites for each user, then average across users
    reach_per_user = np.prod(adjusted_reach_matrix, axis=1)  # Product across websites
    reach = 1 - np.mean(reach_per_user)  # 1 - probability of no exposure

    # Step 6: Compute CTR based on PaC article
    clickthrough = si["Clickthrough"].values  # Clickthrough rates (q_j)
    beta_gamma_q = beta * gamma * clickthrough  # Combined beta, gamma, and clickthrough

    # Compute the term for CTR: (1 - beta_j * gamma_j * q_j) raised to the power of z_ij
    adjusted_ctr_matrix = np.power(
        1 - beta_gamma_q, pvm.values
    )  # (1 - beta_j * gamma_j * q_j) ** z_ij
    adjusted_ctr_matrix = np.clip(
        adjusted_ctr_matrix, 0, 1
    )  # Ensure no negative probabilities

    ctr_per_user = np.prod(adjusted_ctr_matrix, axis=1)
    ctr = 1 - np.mean(ctr_per_user)  # 1 - probability of no click

    return reach, ctr


if __name__ == "__main__":
    # TODO: So far CDE performs better when p >> n and CLasso performs better when n >> p
    # TODO: Script uses 80-20 train-test data, K-fold seems infeasible if considering high dimension
    # TODO: We have yet to consider demographic constraints
    # TODO: CLasso has linear inequality but not sure if its needed

    try:
        # p = 500 so n = 200 represents p >> n (high dim) while n = 1000 represents n >> p (classic)
        sample_sizes = [946 , 2500]

        reach_results = {757: {}, 2000: {}}
        ctr_results = {757: {}, 2000: {}}
        sparsity_results = {757: {}, 2000: {}}

        b_values = np.linspace(10, 500, 20)
        factor = 1.0
        p = 500
        k = 1

        lambda_list_cde = [i / 20 for i in range(19, 0, -1)]
        lambda_list_classo = np.linspace(0.000000000001, 1.0, 11)

        page_view_matrix = pd.read_csv(
            r"ReferencesPaC/Page_View_Matrix_Example.csv", header=0, index_col=0
        )
        site_info = pd.read_csv(r"ReferencesPaC/500_Site_Info_Example.csv", header=0)

        for n in sample_sizes:
            print(f"Sample size: {n}")
            page_view_matrix_subset = page_view_matrix.sample(n=n, random_state=2)

            train_matrix, test_matrix = train_test_split(
                page_view_matrix_subset, test_size=0.2, random_state=2
            )

            Y = train_matrix @ site_info['Clickthrough'].values
            site_info['gamma'] = 1 / (site_info['Pages'] * site_info['Cost'])
            scaling_factors = site_info['gamma'] * site_info['Clickthrough']
            X = train_matrix.values * scaling_factors.values[np.newaxis, :]
            X = pd.DataFrame(X, columns=train_matrix.columns, index=train_matrix.index)
            sigma = 2 * X.T @ X
            eta = 2 * X.T @ Y

            reach_cde, ctr_cde = [], []
            reach_classo, ctr_classo = [], []

            sparsity_cde, sparsity_classo = [], []

            A = np.ones((1, p))

            for b in b_values:
                print(f"Processing Budget: {b}")
                lambda_max, original_factor = find_lambda_max_cplex(
                    sigma, eta, A, [b], p, k, factor
                )

                if lambda_max <= 0:
                    raise Exception("Lambda_max=0!")

                # ===== CDE =====
                best_w_cde = None
                best_reach_cde, best_ctr_cde, best_lambda_cde = 0, 0, None

                for _lambda in lambda_list_cde:
                    w_cde = CDE_DOcplex(sigma, eta, A, [b], p, k, original_factor, _lambda
                                        )

                    if isinstance(w_cde, np.ndarray):
                        r_cde, c_cde = compute_reach_and_ctr(
                            test_matrix, site_info, w_cde
                        )
                        if c_cde > best_ctr_cde:
                            best_reach_cde, best_ctr_cde, best_lambda_cde = (
                                r_cde,
                                c_cde,
                                _lambda,
                            )
                            best_w_cde = w_cde

                reach_cde.append(best_reach_cde)
                ctr_cde.append(best_ctr_cde)
                print(f"Best Lambda for CDE: {best_lambda_cde}")
                print(f"Best reach for CDE: {best_reach_cde}")

                # ===== CLasso =====
                best_w_classo = None
                best_reach_classo, best_ctr_classo, best_lambda_classo = 0, 0, None

                # lambda_max = find_lambda_max_classo(X, Y, A, [b], p, k=1)

                for lambda_value in lambda_list_classo:
                    w_classo = constrained_lasso(
                        X, Y, A, [b], p, k, lambda_value
                    )
                    if isinstance(w_classo, np.ndarray):
                        r_classo, c_classo = compute_reach_and_ctr(
                            test_matrix, site_info, w_classo
                        )
                        if c_classo > best_ctr_classo:
                            best_reach_classo, best_ctr_classo, best_lambda_classo = (
                                r_classo,
                                c_classo,
                                lambda_value,
                            )
                            best_w_classo = w_classo

                reach_classo.append(best_reach_classo)
                ctr_classo.append(best_ctr_classo)
                print(f"Best Lambda for CLasso: {best_lambda_classo}")
                print(f"Best reach for CLasso: {best_reach_classo}")
                # TODO FIND LAMBDA MAX FOR CLASSO RANGE WILE BE TILL 0.1125

                # ===== Sparsity Comparison =====
                if best_w_cde is not None:
                    cde_sparsity = np.sum(np.abs(best_w_cde) > 1e-9)
                else:
                    cde_sparsity = 0

                if best_w_classo is not None:
                    classo_sparsity = np.sum(np.abs(best_w_classo) > 1e-9)
                else:
                    classo_sparsity = 0

                sparsity_cde.append(cde_sparsity)
                sparsity_classo.append(classo_sparsity)

            reach_results[n] = {"CDE": reach_cde, "CLasso": reach_classo}
            ctr_results[n] = {"CDE": ctr_cde, "CLasso": ctr_classo}
            sparsity_results[n] = {"CDE": sparsity_cde, "CLasso": sparsity_classo}

        # results = []
        # for n in sample_sizes:
        #     for i, b in enumerate(b_values):
        #         results.append({
        #             "Sample Size (n)": n,
        #             "Budget": b,
        #             "Reach_CDE": reach_results[n]["CDE"][i],
        #             "CTR_CDE": ctr_results[n]["CDE"][i],
        #             "Sparsity_CDE": sparsity_results[n]["CDE"][i],
        #             "Reach_CLasso": reach_results[n]["CLasso"][i],
        #             "CTR_CLasso": ctr_results[n]["CLasso"][i],
        #             "Sparsity_CLasso": sparsity_results[n]["CLasso"][i]
        #         })
        #
        # results_df = pd.DataFrame(results)
        # results_df.to_csv(f"cde_classo_results_{pd.Timestamp.now():%Y%m%d_%H%M%S}.csv", index=False)
        # print("Results saved to CSV")

        # response = requests.post(
        #     f"https://ntfy.sh/firaz_python",
        #     data="✅ Script finished running".encode("utf-8"),
        #     headers={
        #         "Title": "Script Completed".encode("utf-8").decode("latin-1"),
        #         "Priority": "high",
        #         "Tags": "check",
        #     },
        # )

        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=[
                "Reach (n=757)",
                "Click Rate (n=757)",
                "Sparsity (n=757)",
                "Reach (n=2000)",
                "Click Rate (n=2000)",
                "Sparsity (n=2000)",
            ],
        )

        for i, n in enumerate(sample_sizes, start=1):
            fig.add_trace(
                go.Scatter(
                    x=b_values,
                    y=reach_results[n]["CDE"],
                    mode="lines+markers",
                    name=f"CDE - Reach (n={n})",
                    line=dict(color="red"),
                ),
                row=i,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=b_values,
                    y=reach_results[n]["CLasso"],
                    mode="lines+markers",
                    name=f"CLasso - Reach (n={n})",
                    line=dict(color="black"),
                ),
                row=i,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=b_values,
                    y=ctr_results[n]["CDE"],
                    mode="lines+markers",
                    name=f"CDE - CTR (n={n})",
                    line=dict(color="red"),
                ),
                row=i,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=b_values,
                    y=ctr_results[n]["CLasso"],
                    mode="lines+markers",
                    name=f"CLasso - CTR (n={n})",
                    line=dict(color="black"),
                ),
                row=i,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=b_values,
                    y=sparsity_results[n]["CDE"],
                    mode="lines+markers",
                    name=f"CDE - Sparsity (n={n})",
                    line=dict(color="red"),
                ),
                row=i,
                col=3,
            )
            fig.add_trace(
                go.Scatter(
                    x=b_values,
                    y=sparsity_results[n]["CLasso"],
                    mode="lines+markers",
                    name=f"CLasso - Sparsity (n={n})",
                    line=dict(color="black"),
                ),
                row=i,
                col=3,
            )

        fig.update_layout(
            title_text="CDE and CLasso Performance for Different Sample Sizes (n=757, 2000)",
            height=900,
            showlegend=False,
        )
        fig.update_xaxes(title_text="Budget")
        fig.update_yaxes(title_text="")
        fig.show()


    except Exception as e:
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
        print(f"Error running script: {e}")

    print("Done!")
