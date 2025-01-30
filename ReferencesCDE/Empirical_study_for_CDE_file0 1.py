import numpy as np
import pandas as pd
import scipy
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1)

from portfolio_backtester import *
from CDE_cplex_revised_2 import *

# import os
# os.chdir('/home/dechuan.zhu/backtest_library/')


if __name__ == '__main__':
    list_file=['25_Portfolios_5x5.csv',
     'Industry.csv',
     'international.csv',
     'SPSectors.csv']

    # CDE_DOcplex_with_eta=backtest_model(CDE_DOcplex_with_eta_cv, ['ex_return'],
    #                                               trace_back=True, name='CDE_math', missing_val=True)

    # file_index = 0
    # print(list_file[file_index], '\n------------------------------')
    # data = pd.read_csv(f'portfolio_backtester/data/{list_file[file_index]}', index_col='Date', parse_dates=True)
    #
    # RF = data.RF
    # data = data.drop(columns=['RF'])


    # file_index = 1
    # data = pd.read_csv(f'portfolio_backtester/data/{list_file[file_index]}', index_col='Date', parse_dates=True)
    # RF = data.RF
    # data = data.drop(columns=['RF'])lk,.





    #
    # file_index = 2
    # print(list_file[file_index], '\n------------------------------')
    # data = pd.read_csv(f'portfolio_backtester/data/{list_file[file_index]}', index_col='Date', parse_dates=True)
    # RF = data['T-bill(cont.comp)']
    # data = data.drop(columns=['T-bill(cont.comp)'])


    #
    #

    # file_index = 3
    # print(list_file[file_index], '\n------------------------------')
    # data = pd.read_csv(f'portfolio_backtester/data/{list_file[file_index]}', index_col='Date', parse_dates=True)
    # RF = data['T-bill']
    # data = data.drop(columns=['T-bill'])


    # CDE_DOcplex_with_eta.backtest(data, freq_data='M', freq_strategy='M',
    #
    #                                     window=120, data_type='ex_return', rf=RF)
    # returns = CDE_DOcplex_with_eta.get_net_excess_returns()
    # portfolios = CDE_DOcplex_with_eta.get_portfolios()
    # metrics = {}
    # metrics['sharpe'] = CDE_DOcplex_with_eta.get_sharpe()
    # metrics['turnover'] = CDE_DOcplex_with_eta.get_turnover()
    # metrics['CEQ'] = CDE_DOcplex_with_eta.get_ceq()
    #
    # np.savetxt(f'CDE results/{list_file[file_index]}_returns.csv', returns, delimiter=',')
    # np.savetxt(f'CDE results/{list_file[file_index]}_portfolios.csv', portfolios, delimiter=',')
    # metrics = pd.DataFrame(metrics, index=[list_file[file_index]])
    # metrics.to_csv(f'CDE results/{list_file[file_index]}_metrics.csv')


    #sp100
    # Tbills = fetch_data('T-bills 20020102-20211020.csv')
    # weekly_rf = Tbills['4 weeks'] / 52 / 100 * 4
    # weekly_rf = weekly_rf.resample('D').ffill().fillna(method='ffill')
    # file = 'SP100 20060901-20211015.csv'
    # # file='SP500 20060901-20211015.csv'
    # stoptime = '2021-06-20'
    # print(file, '\n------------------------------')
    # data = fetch_data(file)
    # data = data.loc[:stoptime]
    # data = data.resample('M').last()
    # data = data.pct_change().iloc[1:]
    # data = data.dropna(axis=1)
    # RF = weekly_rf.loc[data.index]

    # sp500
    Tbills = fetch_data('T-bills 20020102-20211020.csv')
    weekly_rf = Tbills['4 weeks'] / 52 / 100 * 4
    weekly_rf = weekly_rf.resample('D').ffill().fillna(method='ffill')
    # file = 'SP100 20060901-20211015.csv'
    file='SP500 20060901-20211015.csv'
    stoptime = '2021-06-20'
    print(file, '\n------------------------------')
    data = fetch_data(file)
    data = data.loc[:stoptime]
    data = data.resample('M').last()
    data = data.pct_change().iloc[1:]
    data = data.dropna(axis=1)
    RF = weekly_rf.loc[data.index]

    window=120
    #
    extra_data = pd.DataFrame(np.hstack((np.zeros(window-1), np.arange(data.shape[0]-window+1) + 1)), index=data.index)



    # CDE_DOcplex_with_eta_2 = backtest_model(CDE_DOcplex_with_eta_cv_2, ['ex_return'],
    #                                     need_extra_data=True, name='CDE_math', missing_val=True)
    #
    # CDE_DOcplex_with_eta_2.backtest(data, freq_data='W', freq_strategy='W',
    #                                     window=120, data_type='price', rf=RF, extra_data=extra_data)

    global counter
    counter=1
    CDE_DOcplex_with_eta_2 = backtest_model(CDE_DOcplex_with_eta_cv_3, ['ex_return'],
                                         name='CDE_math', missing_val=True)

    CDE_DOcplex_with_eta_2.backtest(data, freq_data='M', freq_strategy='M',
                                        window=120, data_type='ex_return', rf=RF)

    returns = CDE_DOcplex_with_eta_2.get_net_excess_returns()
    portfolios = CDE_DOcplex_with_eta_2.get_portfolios()
    metrics = {}
    metrics['sharpe'] = CDE_DOcplex_with_eta_2.get_sharpe()
    metrics['turnover'] = CDE_DOcplex_with_eta_2.get_turnover()
    metrics['CEQ'] = CDE_DOcplex_with_eta_2.get_ceq()

    np.savetxt(f'CDE results/{file}_returns.csv', returns, delimiter=',')
    np.savetxt(f'CDE results/{file}_portfolios.csv', portfolios, delimiter=',')
    metrics = pd.DataFrame(metrics, index=[file])
    metrics.to_csv(f'CDE results/{file}_metrics.csv')







