# Utility functions for the dynamics example

import numpy as np


# Stochasticity arises from both the excitation ground motion and model parameters

def sdof_boucwen_prop(samples, time_vec):
    """ 
    Compute QoI (displacement time-series) for a sdof Bouc-Wen model, both the model parameters and input excitation are random
    
    samples: samples: random field and random parameter vector
    random field (ground motion acceleration time-series [cm.s2]) is ndarray of shape (nsamples, d_excitation)
    random parameter vector is ndarray of shape (N, 5)
    
    time_vec: the sampling times of the excitation - ndarray (d, )
    output is the displacement time-series [cm]
    """
    # Set a fixed parameter value - units are k[cN/cm], r0[cm], delta[], n[], c[cN.s/m]
    params = np.concatenate(samples[0, :4])
    input_acceleration = samples[0, -1]
    # Simulate the behavior of the system forward in time using a fourth order Runge-Kutta method
    ys = np.zeros((3, time_vec.size))
    for i, tn in enumerate(time_vec[:-1]):
        tnp1, ynp1 = one_step_RK4(fun_deriv=deriv_sdof_boucwen, dt=time_vec[i+1]-tn, tn=tn, yn=ys[:, i],
                                  params=params, input_acceleration=input_acceleration, time_vec=time_vec)
        ys[:, i+1] = ynp1
    # Post-process the solver results: extract displacement time series
    time_disp = ys[0, :]
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
