from scipy.optimize import basinhopping
from numpy.random import rand
from collections import OrderedDict
from modulation import *


def drr_norm_measure_30(v: np.ndarray) -> float:                                  # optimization of interpolation points

    retardance1 = 90
    retardance2 = 90

    t1 = v[0:30]
    t2 = v[30:60]

    h = modulation_matrix2(t1[0], t2[0], retardance1, retardance2)

    for q in range(1, len(t1)):
        h = np.vstack((h, modulation_matrix2(t1[q], t2[q], retardance1, retardance2)))

    return np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf)


ratio_min, ratio_max = 2, 10
omega1_min, omega1_max = 1, 10
retardance1_min, retardance1_max = 90, 180
retardance2_min, retardance2_max = 90, 180
determination_min, determination_max = 16, 100
t1_min, t1_max = 0, 360


# Starting sample
pt = np.zeros(60)
for i in range(0, len(pt)):
    pt[i] = t1_min + rand(1) * (t1_max - t1_min)


print(drr_norm_measure_30(pt))
step_size = np.array([1/10, 1/100, 1/1000])
iteration_size = np.array([50, 50, 50])

for i, j in enumerate(step_size):
    result = basinhopping(drr_norm_measure_30,
                          pt,
                          minimizer_kwargs={"method": "L-BFGS-B"},
                          stepsize=step_size[i],
                          niter=iteration_size[i],
                          disp=True)
    pt = result['x']

solution = pt
sa_results = defaultdict()
sa_results_reformat = OrderedDict()
evaluation = drr_norm_measure2(solution)
sa_results[0] = [solution[0:30], solution[30:60], evaluation]

for i in range(0, 30):
    sa_results_reformat[solution[2 + i]] = solution[2 + i + 16]

final_dictionary = OrderedDict(sorted(sa_results_reformat.items()))

with open('result_basinhopping.csv', 'w') as f:
    for key in final_dictionary.keys():
        f.write("%s,%s\n" % (key, final_dictionary[key]))

print('Solution: f(%s) = %.5f' % (solution, evaluation))
