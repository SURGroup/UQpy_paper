# Audrey Olivier
# Utility functions for the dynamics example

import numpy as np
from scipy.integrate import solve_ivp
from functools import partial


# Define the functions for RunModel

def sdof_boucwen_SE(input_acceleration, time_vec, qoi_type='maximum'):
    """ Compute QoI (displacement) for a sdof Bouc-Wen model:
    params are the model parameters,
    force a time series excitation,
    time_vec the sampling times of the excitation,
    disp_qoi is either 'maximum' or 'time_series',
    """
    params = np.array([1.0, 2.0, 4.0, 3.0, 0.02])
    fun = partial(deriv_sdof_boucwen, params=params, input_acceleration=input_acceleration, time_vec=time_vec)
    sol = solve_ivp(fun, y0=[0., 0., 0.], t_span=[time_vec[0], time_vec[-1]], t_eval=time_vec)
    if check_solution(sol, time_vec):   # There is a problem with that run: return None as a qoi
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
    time_vec, input_acceleration = read_elcentro(scale=scale_factor)
    fun = partial(deriv_sdof_boucwen, params=params, input_acceleration=input_acceleration, time_vec=time_vec)
    sol = solve_ivp(fun, y0=[0., 0., 0.], t_span=[time_vec[0], time_vec[-1]], t_eval=time_vec)
    if check_solution(sol, time_vec):   # There is a problem with that run: return None as a qoi
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
    time_vec, input_acceleration = read_elcentro(scale=scale_factor)
    fun = partial(deriv_sdof_elastoplastic, params=params, input_acceleration=input_acceleration, time_vec=time_vec)
    sol = solve_ivp(fun, y0=[0., 0., 0.], t_span=[time_vec[0], time_vec[-1]], t_eval=time_vec)
    if check_solution(sol, force_data[:, 0]):   # There is a problem with that run
        return None
    return sol.y[0, :]  # return QoI = displacement time series


def sdof_linear_RP(params, scale_factor=1.):
    """ Compute QoI (displacement) for an elastoplastic sdof system. The random variable is params=[k, x_n, c].
     RP stands for random parameters. """
    time_vec, input_acceleration = read_elcentro(scale=scale_factor)
    fun = partial(deriv_sdof_linear, params=params, input_acceleration=input_acceleration, time_vec=time_vec)
    sol = solve_ivp(fun, y0=[0., 0., 0.], t_span=[time_vec[0], time_vec[-1]], t_eval=time_vec)
    if check_solution(sol, force_data[:, 0]):   # There is a problem with that run
        return None
    return sol.y[0, :]  # return QoI = max displacement


# Functions that compute the derivative of the state vector
def deriv_sdof_linear(t, y, params, input_acceleration, time_vec):
    # params: k, c
    stiff, damp = params[0], params[1]
    ft = np.interp(t, time_vec, -1. * input_acceleration)
    return [y[1], ft - stiff * y[0] - damp * y[1]]


def deriv_sdof_elastoplastic(t, y, params, input_acceleration, time_vec):
    # params: k, x_plastic, c
    ft = np.interp(t, time_vec, -1. * input_acceleration)
    stiff, x_plastic, damp = params[0], params[1], params[2]
    if y[0] >= x_plastic:
        stiff = 0.
    return [y[1], ft - stiff * y[0] - damp * y[1]]


def deriv_sdof_boucwen(t, y, params, input_acceleration, time_vec):
    # params: k, beta+gamma, gamma+beta, n, c (if no c, no damping)
    stiff, n_bw, damp = params[0], params[3], 0.
    if len(params) == 5:
        damp = params[4]
    beta_bw, gamma_bw = (params[1] + params[2]) / 2, (params[1] - params[2]) / 2
    ft = np.interp(t, time_vec, -1. * input_acceleration)
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
    elcentro_data = np.array(elcentro_data).T
    time_vec = elcentro_data[0]
    accel = 9.8 * elcentro_data[1]    # the data is provided in [g]
    return time_vec, accel

