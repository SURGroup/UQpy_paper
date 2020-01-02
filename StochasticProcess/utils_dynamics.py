# Utility functions for the dynamics example

import numpy as np


# Part a: stochasticity arises from excitation ground motion

def sdof_boucwen_SE(samples, time_vec):
    """ Compute QoI (displacement time-series) for a sdof Bouc-Wen model, model parameters are fixed.
    samples: samples - ndarray (N x d) - of the ground motion acceleration time-series [cm.s2]
    time_vec: the sampling times of the excitation - ndarray (d, )
    output is the displacement time-series [cm]
    """
    # Set a fixed parameter value - units are k[cN/cm], r0[cm], delta[], n[], c[cN.s/m]
    params = np.array([1.0, 2., 0.9, 3., 2.])
    # Simulate the behavior of the system forward in time using a fourth order Runge-Kutta method
    ys = np.zeros((3, time_vec.size))
    for i, tn in enumerate(time_vec[:-1]):
        tnp1, ynp1 = one_step_RK4(fun_deriv=deriv_sdof_boucwen, dt=time_vec[i+1]-tn, tn=tn, yn=ys[:, i],
                                  params=params, input_acceleration=samples, time_vec=time_vec)
        ys[:, i+1] = ynp1
    # Post-process the solver results: extract displacement time series
    time_disp = ys[0, :]
    return time_disp


# Part b: stochasticity arises from system parameters

def sdof_boucwen_RP(samples, scale_factor=1., return_Rf=False):
    """ Compute QoI (displacement time-series, and possibly restoring force) for a sdof Bouc-Wen model
    samples are samples of the model parameters - the units should be k[N/m], r0[cm], delta[], n[], c[cN.s/m]
    scale_factor indicates the scale of the excitation,
    return_Rf indicates whether to return the restoring force along with the displacement time series as a QoI
    """
    # Read input acceleration from el-centro data set
    time_vec, input_acceleration = read_elcentro(scale=scale_factor)
    # Simulate the behavior of the system forward in time
    ys = np.zeros((3, time_vec.size))
    for i, tn in enumerate(time_vec[:-1]):
        tnp1, ynp1 = one_step_RK4(fun_deriv=deriv_sdof_boucwen, dt=time_vec[i+1]-tn, tn=tn, yn=ys[:, i],
                                  params=samples, input_acceleration=input_acceleration, time_vec=time_vec)
        ys[:, i+1] = ynp1
    # Post-process the solver results: extract displacement and reaction force time series
    time_disp = ys[0, :]    # displacement time series
    time_rf = samples[0] * ys[2, :]    # reaction force
    if return_Rf:
        return [time_disp, time_rf]
    return time_disp


# Helper functions

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

s = 0.5
mu = 0.5
(np.exp(s**2) - 1) * np.exp(2*mu + s**2)