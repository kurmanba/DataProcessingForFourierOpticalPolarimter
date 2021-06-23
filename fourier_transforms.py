import numpy as np


def fourier_coeffs(f: np.ndarray,
                   return_complex=False):
    """
    Calculates Fourier series coefficients of a periodic function.
    Given a periodic, function f(t) with period T. coefficients:
    a0, {a1,a2,...},{b1,b2,...} are calculated such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    Parameters:
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeffs.

    Returns:
    -------
    if return_complex == False, the function returns:

    a0 : coefficient
    a,b : coeffs describing harmonics of the cosines and sines.
    c : complex-valued coeffs if return_complex=True
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

    f = [(a_k[i - 1] * np.cos(2 * np.pi * i * t / period) + b_k[i - 1]
          * np.sin(2 * np.pi * i * t / period)) for i in range(1, order+1)]
    f = a_0 / 2 + sum(f)

    return f


def f_series2(x: np.ndarray,
              t: any) -> float:
    """
    Same as fourier series except designed for optimization tasks.

    Parameters:
    ________
    x: Variable vectors for objective function (coefficient)
    t: instance of function evaluation
    period: Period of periodic signal

    Return:
    ______
    S: f(t) Evaluations of associated function
    """
    # Interpretation of optimization parameters
    a_0, a_k, b_k  = x[0], x[1:5], x[5:9]
    freq = x[9]
    phi = 0
    # function evaluation
    f = [(a_k[i - 1] * np.cos(2 * np.pi * i * t * freq + phi) + b_k[i - 1]
          * np.sin(2 * np.pi * i * t * freq + phi)) for i in range(1, len(a_k) + 1)]
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

