import numpy as np


def whittaker_shannon_interpolation(x: np.ndarray,
                                    xp: np.ndarray,
                                    fp: np.ndarray) -> np.ndarray:
    """
    Function uses the Whittaker-Shannon interpolation to reconstruct
    signal at requested instance according to sampling theory of Shannon.

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

    v = (xp - u.T) / (xp[1] - xp[0])
    m = fp * np.sinc(v)
    f_x = np.sum(m, axis=1)

    return f_x


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
