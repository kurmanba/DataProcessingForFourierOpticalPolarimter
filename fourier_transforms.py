import numpy as np
import pyswarms as ps  # optimizers
from pyswarms.utils.functions import single_obj as fx


def fourier_coeffs(f: np.ndarray,
                   return_complex=False):

    """
    Calculates the first 2*N+1 Fourier series coefficients of a periodic function.

    Given a periodic, function f(t) with period T. coefficients:
    a0, {a1,a2,...},{b1,b2,...} are calculated such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True complex coefficients are returned:
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)
    where we define c_{-n} = complex_conjugate(c_{n})

    Parameters:
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns:
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.
    if return_complex == True, the function returns:
    c : numpy 1-dimensional complex-valued array of size N+1
    """

    y = np.fft.rfft(f) / f.size
    if return_complex:
        return y
    else:
        y *= 2

    a_0 = y[0].real
    a_k = y[1:-1].real
    b_k = -y[1:-1].imag

    return a_0, a_k, b_k


def f_series(a_0: float,
             a_k: np.ndarray,
             b_k: np.ndarray,
             t: float,
             order: int,
             period: float):

    """
    Evaluation of the associated function at instance t from fourier series.
    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    Parameters:
    ________
    a_0: coefficient
    a_k: coefficients
    b_k: coefficients
    t: instance of function evaluation
    period: Period of periodic signal

    Return:
    ______
    S: f(t) Evaluation of associated function
    """
    f = [(a_k[i] * np.cos(2 * np.pi * i * t / period) + b_k[i]
          * np.sin(2 * np.pi * i * t / period)) for i in range(0, order)]
    f = a_0 / 2 + sum(f)

    return f


class LagrangePoly:

    def __init__(self,
                 x,
                 y):
        self.n = len(x)
        self.x = np.array(x)
        self.y = np.array(y)

    def basis(self,
              x,
              j):
        b = [(x - self.x[m]) / (self.x[j] - self.x[m])
             for m in range(self.n) if m != j]
        return np.prod(b, axis=0) * self.y[j]

    def interpolate(self,
                    x):
        b = [self.basis(x, j) for j in range(self.n)]
        return np.sum(b, axis=0)


def f_series2(x: np.ndarray,
              t: any) -> float:

    """
    Evaluation of the associated function at instances t from fourier series.
    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    Parameters:
    ________
    x: coefficient for least square fit
    t: instance of function evaluation
    period: Period of periodic signal

    Return:
    ______
    S: f(t) Evaluations of associated function
    """
    # Interpretation of optimization parameters
    a_0, a_k, b_k  = x[0], x[1:5], x[5:9]
    a_phi, b_phi = x[9:13], x[13:17]
    freqs_a, freqs_b = x[17:21], x[21:25]
    # function evaluation
    f = [(a_k[i - 1] * np.cos(2 * np.pi * i * t * freqs_a[0] + a_phi[i - 1]) + b_k[i - 1]
          * np.sin(2 * np.pi * i * t * freqs_a[0] + b_phi[i - 1])) for i in range(1, len(a_k) + 1)]
    f = a_0 / 2 + sum(f)

    return f


def f_residual(x: np.ndarray,
               t: np.ndarray,
               d: np.ndarray) -> np.ndarray:

    """
    Parameters:
    _________
    x: np.ndarray optimization parameters for the optimizer
    t: instances to be evaluated with associated function
    d: measured data (real)

    Return:
    ______
    residual: residual value of estimation vs model
    """
    residual = d - f_series2(x, t)
    return residual


def f_annealing(x: np.ndarray,
                t: np.ndarray,
                d: np.ndarray) -> float:

    """
    Parameters:
    _________
    x: np.ndarray optimization parameters for the optimizer
    t: instances to be evaluated with associated function
    d: measured data (real)

    Return:
    ______
    residual: residual value of estimation vs model
    """

    residual = float(sum((d - f_series2(x, t))**2))

    return residual

