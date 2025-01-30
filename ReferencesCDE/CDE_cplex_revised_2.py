import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def estimate_para(data,true_eta=False):
    sigma = data.cov(ddof=1).values
    p = sigma.shape[0]
    if true_eta:
        eta = np.zeros(p)
        eta[:10] = 0.3/252
    else:
        eta = data.mean().values
    A = np.ones((1, p))
    # A=np.array([[i+1 for i in range(p)]])
    b, k = 1, 1

    return sigma, eta, p, A, b, k


from docplex.mp.model import Model

def initial_find_lambda_max_cplex(sigma, eta, A, b, p, k, factor):
    # lp1
    model = Model('lp1')
    model.parameters.read.scale = -1
    # model.parameters.barrier.convergetol = 1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4

    # define variables
    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))

    # set objective
    expr = model.sum([model.abs(w[i]) for i in range(p)])
    model.minimize(expr)

    # add constraints
    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    # solve and get the results
    solution = model.solve()

    if model.solve_status.value == 2:
        temp_w=np.array(solution.get_values(w))
        model.clear()
    else:
        model.clear()
        print('Infeasible lp1')


    # lp2
    # variables
    model = Model('lp2')
    model.parameters.read.scale = -1
    # #
    # model.parameters.barrier.convergetol = 1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4

    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))
    _lambda_scaled = model.continuous_var(name='lambda')

    # objective
    model.minimize(_lambda_scaled)

    # constraints
    for row in range(p):
        expr = factor * (temp_w @ sigma[row] +
                         model.dot(gamma, A.T[row])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (temp_w @ sigma[row] +
                         model.dot(gamma, A.T[row])) + _lambda_scaled
        model.add_constraint(expr >= 0)


    # solve and get results
    solution = model.solve()

    if model.solve_status.value == 2:
        lambda_max = solution.get_value(_lambda_scaled)
        if lambda_max != 0:
            new_factor = 1 / lambda_max * factor
        else:
            new_factor = factor

        model.clear()
    else:
        model.clear()
        return 'infeasible lp2'

    return lambda_max, new_factor

def find_lambda_max_cplex(sigma, eta, A, b, p, k, factor):
    # lp1
    model = Model('lp1')
    model.parameters.read.scale = -1
    # model.parameters.barrier.convergetol = 1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4

    # define variables
    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))

    # set objective
    expr = model.sum([model.abs(w[i]) for i in range(p)])
    model.minimize(expr)

    # add constraints
    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    # solve and get the results
    solution = model.solve()

    if model.solve_status.value == 2:
        lp1_norm = solution.get_objective_value()
        model.clear()
    else:
        model.clear()
        print('Infeasible lp1')

    # lp2
    # variables
    model = Model('lp2')
    model.parameters.read.scale = -1
    # #
    # model.parameters.barrier.convergetol = 1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4

    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))
    _lambda_scaled = model.continuous_var(name='lambda')

    # objective
    model.minimize(_lambda_scaled)

    # constraints
    for row in range(p):
        expr = factor * (model.dot(w, sigma[row]) -eta[row] +
                         model.dot(gamma, A.T[row])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (model.dot(w, sigma[row]) -eta[row] +
                         model.dot(gamma, A.T[row])) + _lambda_scaled
        model.add_constraint(expr >= 0)

    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    expr = model.sum([model.abs(w[i]) for i in range(p)])
    model.add_constraint(expr == lp1_norm)

    # solve and get results
    solution = model.solve()

    if model.solve_status.value == 2:
        lambda_max = solution.get_value(_lambda_scaled)
        if lambda_max != 0:
            new_factor = 1 / lambda_max * factor
        else:
            new_factor = factor

        model.clear()
    else:
        model.clear()
        return 'infeasible lp2'

    return lambda_max, new_factor


def CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda_scaled):
    # Create DOcplex model
    model = Model(name='phase 1')

    model.parameters.read.scale = -1
    # model.parameters.barrier.convergetol=1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4

    # Define variables
    # w_plus = np.array([model.continuous_var(name=f'w_plus{i}', lb=0) for i in range(p)])
    # w_minus = np.array([model.continuous_var(name=f'w_minus{i}', lb=0) for i in range(p)])
    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))

    # Set objective
    expr = model.sum(model.abs(w[i]) for i in range(p))
    model.minimize(expr)

    # Add constraints
    for row in range(p):
        expr = factor * (model.dot(w, sigma[row]) -eta[row] +
                         model.dot(gamma, A.T[row])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (model.dot(w, sigma[row]) -eta[row] +
                         model.dot(gamma, A.T[row])) + _lambda_scaled
        model.add_constraint(expr >= 0)

    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    # Solve the problem
    solution = model.solve()

    if model.solve_status.value == 2:
        initial_w = np.array(solution.get_values(w))
        model.clear()
    else:
        model.clear()
        return 'Infeasible CDE_phase 1'

    return initial_w


def CDE_DOcplex_phase2_with_control(sigma,eta,A,b,p,k,factor,_lambda_scaled,target_norm, benchmark_w):
    model = Model(name='phase 2')

    # Increase tolerance on feasilibity
    model.parameters.read.scale = -1
    # model.parameters.barrier.convergetol = 1e-3
    # model.parameters.simplex.tolerances.feasibility = 1e-3
    model.parameters.lpmethod = 4
    # model.parameters.preprocessing.qtolin = 0

    w = np.array(model.continuous_var_list([f'w{i}' for i in range(p)], lb=-model.infinity))
    gamma = np.array(model.continuous_var_list([f'gamma{i}' for i in range(k)], lb=-model.infinity))

    # perturb objective functions
    # w = w_p - w_m
    l1_norm=model.sum(model.abs(benchmark_w[i] - w[i]) for i in range(p))
    model.minimize(l1_norm)

    # Add constraints
    for row in range(p):
        expr = factor * (model.dot(w, sigma[row]) -eta[row] +
                         model.dot(gamma, A.T[row])) - _lambda_scaled
        model.add_constraint(expr <= 0)

        expr = factor * (model.dot(w, sigma[row]) -eta[row] +
                         model.dot(gamma, A.T[row])) + _lambda_scaled
        model.add_constraint(expr >= 0)

    for row in range(k):
        expr = model.dot(w, A[row])
        model.add_constraint(expr == b)

    expr = model.sum(model.abs(w[i]) for i in range(p))
    model.add_constraint(expr == target_norm)

    # Solve the problem
    solution = model.solve()

    if model.solve_status.value == 2:
        enumerate_w = np.array(solution.get_values(w))
        model.clear()
    else:
        # print(model.solve_status)
        # raise Exception('Infeasbiel phase 2 (quadratic)')
        # print('Infeasible phase 2 (enumeration)')
        print(model.solve_status)
        model.clear()
        return 'Infeasible CDE_phase 2'

    return enumerate_w


def CDE_DOcplex_with_control(sigma, eta, A, b, p, k, factor, _lambda_scaled, benchmark_w):
    # phase 1 (default)
    initial_w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda_scaled)
    if type(initial_w) == str:
        raise Exception('Infeasible CDE phase 1')

    # phase 2, change c to var and minimize
    target_norm = np.sum(np.abs(initial_w))
    temp_w = CDE_DOcplex_phase2_with_control(sigma, eta, A, b, p, k, factor, _lambda_scaled, target_norm,
                                                 benchmark_w)
    if type(temp_w) !=str:
        return ('feasible', temp_w)
    else:
        print('infeasible phase 2 (control), return initial w')
        return ('feasible', initial_w)


def CDE_DOcplex_simulation_with_control(list_df, _lambda, benchmark_w):   #no cross-validation, for simulation only
    # Start of the strategy
    data = list_df[0]
    position_nan = data.isna().any().values
    portfolio = np.zeros(data.shape[1])
    data = data[data.columns[position_nan == False]]

    # scale each dataset such that lambda_max is always 1
    sigma, eta, p, A, b, k = estimate_para(data)
    factor = 1 / sigma.diagonal().min()
    lambda_max, original_factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    # original_factor scales lambda_max to 1
    lambda_max = 1

    flag, w = CDE_DOcplex_with_control(sigma, eta, A, b, p, k, original_factor, _lambda, benchmark_w)

    portfolio[position_nan == False] = w

    return portfolio



def CDE_DOcplex_simulation_with_eta(list_df, _lambda, true_eta):
    data=list_df[0]
    sigma,ignored_eta,p,A,b,k = estimate_para(data)        #we do not use eta get from estimate_para function

    # factor=1/sigma.diagonal().min()\
    factor=1
    lambda_max, original_factor = find_lambda_max_cplex(sigma, true_eta, A, b, p, k, factor)
    if lambda_max>0:
    # print(find_lambda_max_cplex(sigma, true_eta, A, b, p, k, original_factor))

        w = CDE_DOcplex_phase1(sigma, true_eta, A, b, p, k, original_factor, _lambda)
    else:
        print('lambda_max=0, skip')

    return w



def CDE_DOcplex_find_sparse_w(list_df):
    # Start of the strategy
    data = list_df[0]
    position_nan = data.isna().any().values
    portfolio = np.zeros(data.shape[1])
    data = data[data.columns[position_nan == False]]

    # scale each dataset such that lambda_max is always 1
    sigma, eta, p, A, b, k = estimate_para(data)
    factor = 1 / sigma.diagonal().min()
    lambda_max, original_factor = initial_find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    # original_factor scales lambda_max to 1
    lambda_max = 1

    w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, original_factor, 0.028)

    portfolio[position_nan == False] = w

    return portfolio


def generate_eta(mean_return,p):           # used for empirical study, not in simulation
    # sparsity=max(int(p*0.5),2)
    # sparsity=5
    # indices=np.argsort(mean_return)[-sparsity:]
    # eta=np.zeros(p)
    #
    # eta[indices]=mean_return[indices].mean()
    # eta[indices] = 0.01

    eta=mean_return
    eta[eta<=0.00]=0


# test version , fix eta
#     sparsity=max(int(p*0.1),2)
#     eta=np.zeros(p)
#     eta[:sparsity]=0.02
#     eta=np.random.normal(0,0.015,p)

    return eta



from sklearn.model_selection import KFold
def CDE_DOcplex_with_eta_cv(list_df,scoring='sharpe'):
    # Start of the strategy
    data = list_df[0]
    position_nan = data.isna().any().values
    portfolio = np.zeros(data.shape[1])
    data = data[data.columns[position_nan == False]]


    # lambda_max, original_factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    # # original_factor scales lambda_max to 1
    # lambda_max = 1

    # find lambda_min of data_train for each Fold, decide the lambda_list for whole test
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(data)
    train_index_cv, test_index_cv = [], []

    for train_index, test_index in kf.split(data):
        data_train = data.iloc[train_index, :]
        train_index_cv.append(train_index)
        test_index_cv.append(test_index)



    lambda_list=[i/20 for i in range(19,0,-1)]
    # lambda_list = [i / 20 for i in range(10, 0, -1)]
    # get the score for each lambda candidate
    cv_score = np.zeros(len(lambda_list))
    for index, _lambda in enumerate(lambda_list):
        # print('lambda:', _lambda)
        lambda_test_returns = []
        for train_index, test_index in zip(train_index_cv, test_index_cv):
            data_train, data_test = data.iloc[train_index, :], data.iloc[test_index, :]

            sigma, beta, p, A, b, k = estimate_para(data_train)
            # Different batch of data, different factor to scale lambda_max to 1
            factor = 1
            eta = generate_eta(beta, p)
            temp_lambda_max, factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
            if temp_lambda_max==0:
                print('Lambda_max==0')
            # Only need the factor here.

            w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda)
            if type(w)==str:
                raise Exception('Infeasible CDE_phase 1')
            returns = np.dot(data_test, w)
            lambda_test_returns.extend(returns)


        if scoring == 'sharpe':
            cv_score[index] = np.mean(lambda_test_returns) / np.std(lambda_test_returns, ddof=1)
        else:
            cv_score[index] = np.var(lambda_test_returns)

    # find the lambda that has the best score
    if scoring == 'sharpe':
        _lambda = lambda_list[np.argmax(cv_score)]
    else:
        _lambda = lambda_list[np.argmin(cv_score)]

    # fit the original data with found _lambda
    sigma, beta, p, A, b, k = estimate_para(data)
    factor=1
    eta = generate_eta(beta, p)
    temp_lambda_max, factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda)

    portfolio[position_nan == False] = w
    return portfolio


def CDE_DOcplex_with_eta_cv_2(list_df,extra_data,scoring='sharpe'):
    # only change eta after 12 periods each time. period is tracked by extra_data
    # Start of the strategy
    global last_eta

    if extra_data.iloc[-1].values[0] % 12 ==1:
        need_new_eta=True
    else:
        need_new_eta=False


    data = list_df[0]
    position_nan = data.isna().any().values
    portfolio = np.zeros(data.shape[1])
    data = data[data.columns[position_nan == False]]


    # lambda_max, original_factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    # # original_factor scales lambda_max to 1
    # lambda_max = 1

    # find lambda_min of data_train for each Fold, decide the lambda_list for whole test
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(data)
    train_index_cv, test_index_cv = [], []

    for train_index, test_index in kf.split(data):
        data_train = data.iloc[train_index, :]
        train_index_cv.append(train_index)
        test_index_cv.append(test_index)



    lambda_list=[i/20 for i in range(19,0,-1)]
    # lambda_list = [i / 20 for i in range(10, 0, -1)]
    # get the score for each lambda candidate
    cv_score = np.zeros(len(lambda_list))
    for index, _lambda in enumerate(lambda_list):
        # print('lambda:', _lambda)
        lambda_test_returns = []
        for train_index, test_index in zip(train_index_cv, test_index_cv):
            data_train, data_test = data.iloc[train_index, :], data.iloc[test_index, :]

            sigma, beta, p, A, b, k = estimate_para(data_train)
            # Different batch of data, different factor to scale lambda_max to 1
            factor = 1

            if need_new_eta:
                eta = generate_eta(beta, p)
            else:
                eta = last_eta

            temp_lambda_max, factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
            if temp_lambda_max==0:
                print('Lambda_max==0')
            # Only need the factor here.

            w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda)
            if type(w)==str:
                raise Exception('Infeasible CDE_phase 1')
            returns = np.dot(data_test, w)
            lambda_test_returns.extend(returns)


        # if scoring == 'sharpe':
        #     cv_score[index] = np.mean(lambda_test_returns) / np.std(lambda_test_returns, ddof=1)
        # else:
        #     cv_score[index] = np.var(lambda_test_returns)
    cv_score[index] = np.var(lambda_test_returns)
    # find the lambda that has the best score
    _lambda = lambda_list[np.argmin(cv_score)]

    # if scoring == 'sharpe':
    #     _lambda = lambda_list[np.argmax(cv_score)]
    # else:
    #     _lambda = lambda_list[np.argmin(cv_score)]

    # fit the original data with found _lambda
    sigma, beta, p, A, b, k = estimate_para(data)
    factor=1

    if need_new_eta:
        eta = generate_eta(beta, p)
    else:
        eta = last_eta

    temp_lambda_max, factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda)

    portfolio[position_nan == False] = w
    last_eta = eta
    print('lanmbda:',_lambda)
    print(eta)
    print('-----------------')
    return portfolio


def CDE_DOcplex_with_eta_cv_3(list_df,scoring='sharpe'):
    # only change eta after 12 periods each time. period is tracked by extra_data
    # Start of the strategy
    global last_eta, counter

    try:
        counter
    except NameError:
        counter=1

    if counter % 12 ==1:
        need_new_eta=True
    else:
        need_new_eta=False


    data = list_df[0]
    position_nan = data.isna().any().values
    portfolio = np.zeros(data.shape[1])
    data = data[data.columns[position_nan == False]]


    # lambda_max, original_factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    # # original_factor scales lambda_max to 1
    # lambda_max = 1

    # find lambda_min of data_train for each Fold, decide the lambda_list for whole test
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(data)
    train_index_cv, test_index_cv = [], []

    for train_index, test_index in kf.split(data):
        data_train = data.iloc[train_index, :]
        train_index_cv.append(train_index)
        test_index_cv.append(test_index)



    lambda_list=[i/20 for i in range(19,8,-1)]
    # lambda_list = [i / 20 for i in range(10, 0, -1)]
    # get the score for each lambda candidate
    cv_score = np.zeros(len(lambda_list))
    for index, _lambda in enumerate(lambda_list):
        # print('lambda:', _lambda)
        lambda_test_returns = []
        for train_index, test_index in zip(train_index_cv, test_index_cv):
            data_train, data_test = data.iloc[train_index, :], data.iloc[test_index, :]

            sigma, beta, p, A, b, k = estimate_para(data_train)
            # Different batch of data, different factor to scale lambda_max to 1
            factor = 1

            if need_new_eta:
                eta = generate_eta(beta, p)
            else:
                eta = last_eta

            temp_lambda_max, factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
            if temp_lambda_max==0:
                print('Lambda_max==0')
            # Only need the factor here.

            w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda)
            if type(w)==str:
                raise Exception('Infeasible CDE_phase 1')
            returns = np.dot(data_test, w)
            lambda_test_returns.extend(returns)
        print(_lambda)


        # if scoring == 'sharpe':
        #     cv_score[index] = np.mean(lambda_test_returns) / np.std(lambda_test_returns, ddof=1)
        # else:
        #     cv_score[index] = np.var(lambda_test_returns)
    cv_score[index] = np.var(lambda_test_returns)
    # find the lambda that has the best score
    _lambda = lambda_list[np.argmin(cv_score)]

    # if scoring == 'sharpe':
    #     _lambda = lambda_list[np.argmax(cv_score)]
    # else:
    #     _lambda = lambda_list[np.argmin(cv_score)]

    # fit the original data with found _lambda
    sigma, beta, p, A, b, k = estimate_para(data)
    factor=1

    if need_new_eta:
        eta = generate_eta(beta, p)
    else:
        eta = last_eta

    temp_lambda_max, factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
    w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda)

    portfolio[position_nan == False] = w
    last_eta = eta
    counter+=1
    print('lanmbda:',_lambda)
    print(eta)
    print(counter)
    print('-----------------')
    return portfolio


def CDE_DOcplex_with_eta_cv_iterateeta(list_df, scoring='sharpe'):
    # Monte-Carlo method where we iterate over eta generated from zero-mean multivariate normal distribution, and take average over w's
    # Start of the strategy
    data = list_df[0]
    position_nan = data.isna().any().values
    portfolio = np.zeros(data.shape[1])
    data = data[data.columns[position_nan == False]]
    p=data.shape[1]
    iteration = 10

    def one_iteration(data,eta):
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(data)
        train_index_cv, test_index_cv = [], []

        for train_index, test_index in kf.split(data):
            data_train = data.iloc[train_index, :]
            train_index_cv.append(train_index)
            test_index_cv.append(test_index)

        lambda_list = [i / 20 for i in range(19, 0, -1)]
        # lambda_list = [i / 20 for i in range(10, 0, -1)]
        # get the score for each lambda candidate
        cv_score = np.zeros(len(lambda_list))
        for index, _lambda in enumerate(lambda_list):
            # print('lambda:', _lambda)
            lambda_test_returns = []
            for train_index, test_index in zip(train_index_cv, test_index_cv):
                data_train, data_test = data.iloc[train_index, :], data.iloc[test_index, :]

                sigma, beta, p, A, b, k = estimate_para(data_train)
                # Different batch of data, different factor to scale lambda_max to 1
                factor = 1
                # eta = generate_eta(beta, p)
                temp_lambda_max, factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
                if temp_lambda_max == 0:
                    print('Lambda_max==0')
                # Only need the factor here.

                w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda)
                if type(w) == str:
                    raise Exception('Infeasible CDE_phase 1')
                returns = np.dot(data_test, w)
                lambda_test_returns.extend(returns)

            if scoring == 'sharpe':
                cv_score[index] = np.mean(lambda_test_returns) / np.std(lambda_test_returns, ddof=1)
            else:
                cv_score[index] = np.var(lambda_test_returns)

        # find the lambda that has the best score
        if scoring == 'sharpe':
            _lambda = lambda_list[np.argmax(cv_score)]
        else:
            _lambda = lambda_list[np.argmin(cv_score)]

        # fit the original data with found _lambda
        sigma, beta, p, A, b, k = estimate_para(data)
        factor = 1
        # eta = generate_eta(beta, p)
        temp_lambda_max, factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
        w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, factor, _lambda)

        return w

    w=np.zeros(p)
    for i in range(iteration):
        eta=np.random.normal(0,0.015,p)
        w = w + one_iteration(data, eta)
    w=w/iteration
    portfolio[position_nan == False]=w

    return portfolio


if __name__ == '__main__':
    # list_file = ['25_Portfolios_5x5.csv',
    #              'Industry.csv',
    #              'international.csv',
    #              'SPSectors.csv']
    #
    # file_index = 1
    # data = pd.read_csv(f'portfolio_backtester/data/{list_file[file_index]}', index_col='Date', parse_dates=True)
    # RF = data.RF
    # data = data.drop(columns=['RF'])
    # window=120
    #
    # data_train=data.iloc[:window]
    # data_test=data.iloc[window]
    # w=CDE_DOcplex_with_eta_cv([data_train])


    # p,n,model=400,126*4+1,'1'
    # p,n,model=100, 127, '2'
    # model='1'
    # p,n=600,252*3+1
    model='2'
    # p,n=100,127
    # p,n=200,126*2+1
    # p,n=400, 252*2+1
    p, n = 600, 252 * 3 + 1
    #
    result_dic = {}
    portfolio_dic = {}
    lambda_list = [i / 20 for i in range(19, 0, -1)]
    for _lambda in lambda_list:
        portfolio_dic[_lambda] = []

    for counter in range(100):
        data = pd.read_csv(f'simulation datasets/{model} simulation data {n}x{p} {counter + 1}.csv',
                           index_col='Date',
                           parse_dates=True)
        sigma, eta, p, A, b, k = estimate_para(data.iloc[:-1], true_eta=True)
        factor = 1
        lambda_max, original_factor = find_lambda_max_cplex(sigma, eta, A, b, p, k, factor)
        if lambda_max<=0:
            raise Exception('Lambda_max=0!')

        return_list = np.empty(len(lambda_list))
        for index, _lambda in enumerate(lambda_list):
            print(f'Optimizing {index}:{_lambda}')
            w = CDE_DOcplex_phase1(sigma, eta, A, b, p, k, original_factor, _lambda)

            portfolio_dic[_lambda].append(w)
            return_list[index] = data.iloc[-1].values @ w
        print(f'finished data {counter + 1}')
        result_dic[counter] = return_list
    result = pd.DataFrame.from_dict(result_dic)
    result.index = lambda_list
    result.to_csv(
        f'simulation datasets/results/{model} simulation data {n}x{p}_100_CDE_fixed_lambda_b=1_default_tolerance_trueeta__results.csv')
    portfolios = pd.DataFrame.from_dict(portfolio_dic, orient='index')
    portfolios.to_csv(
        f'simulation datasets/results/{model} simulation data {n}x{p}_100_CDE_fixed_lambda_b=1_default_tolerance_trueeta_portfolios.csv')