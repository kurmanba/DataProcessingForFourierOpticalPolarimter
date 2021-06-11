import numpy as np


def whittaker_shannon_interpolation(x: np.ndarray,
                                    xp: np.ndarray,
                                    fp: np.ndarray,
                                    period: float,
                                    left=None,
                                    right=None):
    """
    Function uses the Whittaker-Shannon interpolation to reconstruct
    signal according to sampling theory of Shannon.

    Parameters:
    __________
    x: signal to be evaluated at f(x).
    xp: sampled array of original signal time domain.
    fp: associated function values for xp.

    Return:
    _______
    f_x: Approximation of f(x) at x
    """

    u = np.resize(x, (len(xp), len(x)))
    print(np.size(u))

    v = (xp - u.T) / (xp[1] - xp[0])
    m = fp * np.sinc(v)
    f_x = np.sum(m, axis=1)

    return f_x
