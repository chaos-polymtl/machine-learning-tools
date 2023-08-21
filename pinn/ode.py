# ============================================================================
# Numerical solver of ODEs using Runge-Kutta method
# Author : Valérie Bibeau, Polytechnique Montréal, 2023
# ============================================================================

# ---------------------------------------------------------------------------
# Library
import numpy as np
# ---------------------------------------------------------------------------

def edo(y, prm):
    """Ordinary differiential equation

    Args:
        y (numpy array): Value of dependant variables
        prm (struct): Kinetic parameters

    Returns:
        numpy array: Right hand side of ODEs
    """

    cA = y[0]
    cB = y[1]
    cC = y[2]
    cD = y[3]

    k1 = prm.k1
    k2 = prm.k2
    k3 = prm.k3
    k4 = prm.k4

    f = np.zeros(4)

    f[0] = - k1 * cA + k2 * cB * cC
    f[1] = + k1 * cA - k2 * cB * cC
    f[2] = + k1 * cA - k2 * cB * cC - k3 * cC + k4 * cD
    f[3] = + k3 * cC - k4 * cD

    return f

def runge_kutta(y0, prm, dt, tf):
    """Runge-Kutta method to solve ODEs

    Args:
        y0 (numpy array): Initial condition
        prm (struct): Kinetic parameters
        dt (float): Time step
        tf (float): Simulation time

    Returns:
        numpy array: Time and results
    """

    t = np.array([0])
    mat_y = np.array([y0])

    while t[-1] < tf:

        k1 = dt * edo(y0, prm)
        k2 = dt * edo(y0+k1/2, prm)
        k3 = dt * edo(y0+k2/2, prm)
        k4 = dt * edo(y0+k3, prm)

        y = y0 + 1/6*(k1 + 2*k2 + 2*k3 + k4)

        mat_y = np.append(mat_y, [y], axis=0)

        y0 = np.copy(y)

        t = np.append(t, [t[-1]+dt], axis=0)

    return t, mat_y
