# Audrey Olivier
# Utility functions for the dynamics example

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial


# Define the functions for RunModel

def sdof_boucwen_SE(force, time_vec, qoi_type='maximum'):
    """ Compute QoI (displacement) for a sdof Bouc-Wen model:
    params are the model parameters,
    force a time series excitation,
    time_vec the sampling times of the excitation,
    disp_qoi is either 'maximum' or 'time_series',
    """
    params = np.array([1.0, 2.0, 4.0, 3.0, 0.02])
    force_data = np.concatenate([time_vec.reshape((-1, 1)), np.array(force).reshape((-1, 1))], axis=1)
    fun = partial(deriv_sdof_boucwen, force_data=force_data, params=params)
    sol = solve_ivp(fun, y0=[0., 0., 0.], t_span=[force_data[0, 0], force_data[-1, 0]], t_eval=force_data[:, 0])
    if check_solution(sol, force_data[:, 0]):   # There is a problem with that run: return None as a qoi
        return None
    if qoi_type.lower() == 'maximum':
        return max(sol.y[0, :])
    elif qoi_type.lower() == 'time_series':
        return sol.y[0, :]
    else:
        raise ValueError('qoi_type should be "maximum" or "time_series"')
        
def sdof_boucwen_RP(params, scale_factor=1., qoi_type='maximum'):
    """ Compute QoI (displacement) for a sdof Bouc-Wen model:
    params are the model parameters,
    force a time series excitation,
    time_vec the sampling times of the excitation,
    disp_qoi is either 'maximum' or 'time_series',
    """
    force_data = read_elcentro(scale=scale_factor)
    fun = partial(deriv_sdof_boucwen, force_data=force_data, params=params)
    sol = solve_ivp(fun, y0=[0., 0., 0.], t_span=[force_data[0, 0], force_data[-1, 0]], t_eval=force_data[:, 0])
    if check_solution(sol, force_data[:, 0]):   # There is a problem with that run: return None as a qoi
        return None
    if qoi_type.lower() == 'maximum':
        return max(sol.y[0, :])
    elif qoi_type.lower() == 'time_series':
        return sol.y[0, :]
    else:
        raise ValueError('qoi_type should be "maximum" or "time_series"')


def sdof_elastoplastic_RP(params, scale_factor=1.):
    """ Compute QoI (displacement) for an elastoplastic sdof system. The random variable is params=[k, x_n, c].
     RP stands for random parameters. """
    force_data = read_elcentro(scale=scale_factor)
    fun = partial(deriv_sdof_elastoplastic, force_data=force_data, params=np.array(params).reshape((-1, )))
    sol = solve_ivp(fun, y0=[0., 0.], t_span=[force_data[0, 0], force_data[-1, 0]], t_eval=force_data[:, 0])
    if check_solution(sol, force_data[:, 0]):   # There is a problem with that run
        return None
    return sol.y[0, :]  # return QoI = max displacement


def sdof_linear_RP_50(params, scale_factor=1.):
    """ Compute QoI (displacement) for an elastoplastic sdof system. The random variable is params=[k, x_n, c].
     RP stands for random parameters. """
    force_data = read_elcentro(scale=scale_factor)
    fun = partial(deriv_sdof_linear, force_data=force_data, params=np.array(params).reshape((-1, )))
    sol = solve_ivp(fun, y0=[0., 0.], t_span=[force_data[0, 0], force_data[-1, 0]], t_eval=force_data[:, 0])
    if check_solution(sol, force_data[:, 0]):   # There is a problem with that run
        return None
    return sol.y[0, :]  # return QoI = max displacement


# Functions that compute the derivative of the state vector
def deriv_sdof_linear(t, y, force_data, params):
    # params: k, c
    stiff, damp = params[0], params[1]
    ft = np.interp(t, force_data[:, 0], force_data[:, 1])
    return [y[1], ft - stiff * y[0] - damp * y[1]]


def deriv_sdof_elastoplastic(t, y, force_data, params):
    # params: k, x_plastic, c
    ft = np.interp(t, force_data[:, 0], force_data[:, 1])
    stiff, x_plastic, damp = params[0], params[1], params[2]
    if y[0] >= x_plastic:
        stiff = 0.
    return [y[1], ft - stiff * y[0] - damp * y[1]]


def deriv_sdof_boucwen(t, y, force_data, params):
    # params: k, beta+gamma, gamma+beta, n, c (if no c, no damping)
    stiff, n_bw, damp = params[0], params[3], 0.
    if len(params) == 5:
        damp = params[4]
    beta_bw, gamma_bw = (params[1] + params[2]) / 2, (params[1] - params[2]) / 2
    ft = np.interp(t, force_data[:, 0], force_data[:, 1])
    rdot = y[1] - beta_bw * np.abs(y[1]) * np.abs(y[2]) ** (n_bw-1) * y[2] - gamma_bw * y[1] * np.abs(y[2]) ** n_bw
    return [y[1], ft - stiff * y[2] - damp * y[1], rdot]


def check_solution(sol, t_eval):
    if sol.y.shape[1] != len(t_eval) or sol.t[-1] != t_eval[-1]:
        return True
    return False


# Read El-Centro data

def read_elcentro(scale=1.):
    elcentro_data = []
    with open('elcentro.txt', 'r') as f:
        for line in f:
            row = line.replace('\n', '\t').split('\t')
            elcentro_data.append([float(row[0]), scale * float(row[1])])
    return np.array(elcentro_data)

