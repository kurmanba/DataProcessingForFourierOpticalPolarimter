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


def _find_m(n):                                                                              # helper function to Padua
    ix = np.r_[1:(n + 1) * (n + 2):2]
    if np.mod(n, 2) == 0:
        n2 = n // 2
        offset = np.array([[0, 1] * n2 + [0, ]] * (n2 + 1))
        ix = ix - offset.ravel(order='F')
    return ix


def padua_points_2(n: int,
                   domain=(0, 360, 0, 360)) -> tuple:
    """
    The function returns the 2D interpolation Padua points depending
    on the specified order.

    Parameters:
    ----------
    n: scalar integer interpolation degree
    domain : rectangle [a,b] x [c,d].

    Return:
    -------
    pad: array of shape (2 x (n+1)*(n+2)/2)
        (pad[0,:], pad[1,: ]) defines the Padua points in the domain
        rectangle [a,b] x [c,d].
    """

    a, b, c, d = domain
    t0 = [np.pi] if n == 0 else np.linspace(0, np.pi, n + 1)
    t1 = np.linspace(0, np.pi, n + 2)
    zn = (a + b + (b - a) * np.cos(t0)) / 2
    zn1 = (c + d + (d - c) * np.cos(t1)) / 2

    pad1, pad2 = np.meshgrid(zn, zn1)
    ix = _find_m(n)

    return pad1.ravel(order='F')[ix], pad2.ravel(order='F')[ix]

