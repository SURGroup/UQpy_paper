# Utility functions for the dynamics example

import numpy as np


# Define the three models (Bouc-Wen, elastoplastic, linear)

def sdof_boucwen_infce(samples, scale_factor=1., return_Rf=False):
    """ Compute QoI (displacement time-series, and possibly restoring force) for a sdof Bouc-Wen model
    samples are samples of the model parameters - the units should be k[N/m], r0[cm], delta[], n[], c[cN.s/m]
    scale_factor indicates the scale of the excitation,
    return_Rf indicates whether to return the restoring force along with the displacement time series as a QoI
    """
    # Read input acceleration from el-centro data set
    time_vec, input_acceleration = read_elcentro(scale=scale_factor)
    # Get the parameters
    params = samples.reshape((-1,))
    # Simulate the behavior of the system forward in time
    ys = np.zeros((3, time_vec.size))
    for i, tn in enumerate(time_vec[:-1]):
        tnp1, ynp1 = one_step_RK4(fun_deriv=deriv_sdof_boucwen, dt=time_vec[i+1]-tn, tn=tn, yn=ys[:, i],
                                  params=params, input_acceleration=input_acceleration, time_vec=time_vec)
        ys[:, i+1] = ynp1
    # Post-process the solver results: extract displacement and reaction force time series
    time_disp = ys[0, :]    # displacement time series
    time_rf = params[0] * ys[2, :]    # reaction force
    if return_Rf:
        return [time_disp, time_rf]
    return time_disp


def sdof_elastoplastic_infce(samples, scale_factor=1., return_Rf=False):
    """ Compute QoI (displacement) for an elastoplastic sdof system. The random variable is params=[k, x_n, c].
     RP stands for random parameters. """
    # Read input acceleration from el-centro data set
    time_vec, input_acceleration = read_elcentro(scale=scale_factor)
    # Get the parameters
    params = samples.reshape((-1,))
    # Simulate the behavior of the system forward using a Runge-Kutta fourth order solver
    ys, time_rf = np.zeros((2, time_vec.size)), np.zeros((time_vec.size, ))
    turn_pt = (0., 0.)
    for i, tn in enumerate(time_vec[:-1]):
        tnp1, ynp1 = one_step_RK4(fun_deriv=deriv_sdof_elastoplastic, dt=time_vec[i+1]-tn, tn=tn, yn=ys[:, i],
                                  params=params, turn_pt=turn_pt, input_acceleration=input_acceleration, time_vec=time_vec)
        ys[:, i+1] = ynp1
        time_rf[i+1] = compute_rf_elastoplastic(ynp1, params, turn_pt)
        if np.sign(ys[1, i+1]) != np.sign(ys[1, i]):    # This was a turning point
            turn_pt = (ys[0, i+1], time_rf[i+1])
    # Post-process the solver results
    time_disp = ys[0, :]    # displacement time series
    if return_Rf:
        return [time_disp, time_rf]
    return time_disp


def sdof_linear_infce(samples, scale_factor=1., return_Rf=False):
    """ Compute QoI (displacement) for a linear elastic sdof system. The random variable is params=[k, c].
     RP stands for random parameters. """
    # Read input acceleration from el-centro data set
    time_vec, input_acceleration = read_elcentro(scale=scale_factor)
    # Get the parameters
    params = samples.reshape((-1,))
    # Simulate the behavior of the system forward using a Runge-Kutta fourth order solver
    ys, time_rf = np.zeros((2, time_vec.size)), np.zeros((time_vec.size, ))
    for i, tn in enumerate(time_vec[:-1]):
        tnp1, ynp1 = one_step_RK4(fun_deriv=deriv_sdof_linear, dt=time_vec[i+1]-tn, tn=tn, yn=ys[:, i],
                                  params=params, input_acceleration=input_acceleration, time_vec=time_vec)
        ys[:, i+1] = ynp1
    # Post-process the solver results
    time_disp = ys[0, :]    # displacement time series
    time_rf = params[0] * ys[0, :]
    if return_Rf:
        return [time_disp, time_rf]
    return time_disp


# Helper functions for Runge-Kutta method

def deriv_sdof_boucwen(t, y, params, input_acceleration, time_vec):
    """ Compute the derivative of the state vector [x, xdot, r] for the Bouc-Wen model """
    # params: k, r0, delta, n, c (if no c, no damping, c=0.)
    stiff, n_bw, damp = params[0], params[3], 0.
    if len(params) == 5:    # damping is non-zero
        damp = params[4]*1e-2
    beta_bw = params[2] / (params[1]**n_bw)
    gamma_bw = beta_bw * (1.-params[2]) / params[2]
    ft = np.interp(t, time_vec, -1. * input_acceleration)
    rdot = y[1] - beta_bw * np.abs(y[1]) * np.abs(y[2]) ** (n_bw-1) * y[2] - gamma_bw * y[1] * np.abs(y[2]) ** n_bw
    return np.array([y[1], ft - stiff * y[2] - damp * y[1], rdot])


def deriv_sdof_elastoplastic(t, y, params, turn_pt, input_acceleration, time_vec):
    """ Compute the derivative of the state vector [x, xdot] for the elastoplastic model """
    # params: k, x_plastic, c
    ft = np.interp(t, time_vec, -1. * input_acceleration)
    stiff, x_plastic, damp = params[0], params[1], params[2]*1e-2
    # Compute restoring force
    fs = compute_rf_elastoplastic(y, params, turn_pt)
    return np.array([y[1], ft - fs - damp * y[1]])


def compute_rf_elastoplastic(y, params, turn_pt):
    # params: k [cN/cm], x_plastic [cm], c [cN.s/m]
    # Transform parameters to their units [cN/cm, cm, cN.s/cm]
    stiff, x_plastic, damp = params[0], params[1], params[2]*1e-2
    xt, Rt = turn_pt[0], turn_pt[1]
    # Compute restoring force
    fs = Rt + stiff * (y[0] - xt)
    if fs > stiff * x_plastic:
        fs = stiff * x_plastic
    elif fs < - stiff * x_plastic:
        fs = - stiff * x_plastic
    return fs


def deriv_sdof_linear(t, y, params, input_acceleration, time_vec):
    """ Compute the derivative of the state vector [x, xdot] for the linear model """
    # params: k, c
    stiff, damp = params[0], params[1]*1e-2
    ft = np.interp(t, time_vec, -1. * input_acceleration)
    return np.array([y[1], ft - stiff * y[0] - damp * y[1]])


def one_step_RK4(fun_deriv, dt, tn, yn, **kwargs):
    """ One step of the fourth-order Runge-Kutta method. """
    k1 = dt * fun_deriv(tn, yn, **kwargs)
    k2 = dt * fun_deriv(tn + 0.5 * dt, yn + 0.5 * k1, **kwargs)
    k3 = dt * fun_deriv(tn + 0.5 * dt, yn + 0.5 * k2, **kwargs)
    k4 = dt * fun_deriv(tn + dt, yn + k3, **kwargs)
    ynp1 = yn + 1/6 * (k1 + 2.*k2 + 2.*k3 + k4)
    return tn+dt, ynp1


def read_elcentro(scale=1., total_time=40):
    """ Read data from the El-Centro earthquake, scale is a scale factor, total_time the end time [s] """
    input_dict = {'S90W': ['USACA47.035.txt', 45, 381]}
    comp = 'S90W'
    dt, freq = 0.02, 50    # dt [s], freq [Hz]
    with open(input_dict[comp][0], 'r') as f:
        accel = []
        for i, line in enumerate(f):
            if (i > input_dict[comp][1]) and (i < input_dict[comp][2]):
                row = line.replace('\n', '').split()
                accel.extend([-scale * float(r) for r in row])
    time_vec = np.linspace(0., len(accel) * dt, len(accel)+1)
    return time_vec[:freq*total_time+1], np.array(accel)[:freq*total_time+1]


# Helper functions to save or load data or results


def generate_synthetic_data(scale, params, namefile=None):
    from UQpy.Distributions import Distribution
    # Read data from El-Centro Earthquake
    time_vec, accel = read_elcentro(scale=scale)
    # Solve ODE forward in time
    response = sdof_boucwen_RP(samples=params, scale_factor=scale, return_Rf=False)
    disp_time_series = np.array(response)
    # Add noise to the displacement time series
    std_noise = 0.05 * np.sqrt(np.mean(disp_time_series ** 2))  # 5% RMS noise
    data_noisy = disp_time_series + Distribution('normal').rvs(
        params=[0., std_noise], nsamples=disp_time_series.size).reshape((-1,))
    if namefile is not None:
        np.savetxt(namefile, data_noisy)
    return data_noisy


def load_synthetic_data(namefile):
    data_noisy = np.loadtxt(namefile)
    std_noise = 0.05 * np.sqrt(np.mean(data_noisy ** 2))  # 5% RMS noise
    return data_noisy, std_noise ** 2


def save_results(namefile, key, value):
    import pickle
    from os import path
    if path.exists(namefile) and path.getsize(namefile) > 0:
        with open(namefile, 'rb') as f:
            results_dict = pickle.load(f)
    else:
        results_dict = {}
    results_dict.update({key: value})
    with open(namefile, 'wb') as f:
        pickle.dump(results_dict, f)


def load_results(namefile, key):
    import pickle
    with open(namefile, 'rb') as f:
        results_dict = pickle.load(f)
    return results_dict[key]