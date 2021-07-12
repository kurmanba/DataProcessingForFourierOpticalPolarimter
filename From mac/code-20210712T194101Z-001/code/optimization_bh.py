import numpy as np
from modulation import drr_norm_measure
from scipy.optimize import basinhopping
from numpy.random import rand


ratio_min, ratio_max = 2, 10
omega1_min, omega1_max = 1, 10
retardance1_min, retardance1_max = 90, 180
retardance2_min, retardance2_max = 90, 180
determination_min, determination_max = 16, 100


# Starting sample
pt = np.zeros(5)
pt[0] = ratio_min + rand(1) * (ratio_max - ratio_min)
pt[1] = omega1_min + rand(1) * (omega1_max - omega1_min)
pt[2] = retardance1_min + rand(1) * (retardance1_max - retardance1_min)
pt[3] = retardance2_min + rand(1) * (retardance2_max - retardance2_min)
pt[4] = determination_min + rand(1) * (determination_max - determination_min)


def r1_min(x):
    return 10 - x[0]


def r1_max(x):
    return x[0] - 2


def om1_min(x):
    return 10 - x[0]


def om1_max(x):
    return x[0] - 1


def ret1_min(x):
    return 180 - x[0]


def ret1_max(x):
    return x[0] - 90


def det1_min(x):
    return 100 - x[0]


def det1_max(x):
    return x[0] - 16


c1 = {"type": "ineq", "fun": r1_min}
c2 = {"type": "ineq", "fun": r1_max}
c3 = {"type": "ineq", "fun": om1_min}
c4 = {"type": "ineq", "fun": om1_max}
c5 = {"type": "ineq", "fun": ret1_min}
c6 = {"type": "ineq", "fun": ret1_max}
c7 = {"type": "ineq", "fun": ret1_min}
c8 = {"type": "ineq", "fun": ret1_max}

minimizer_kwargs = dict(method="COBYLA", constraints=(c1, c2, c3))
result = basinhopping(drr_norm_measure,
                      pt,
                      minimizer_kwargs=minimizer_kwargs,
                      stepsize=0.5,
                      niter=1000,
                      disp=True)
solution = result['x']
evaluation = drr_norm_measure(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
