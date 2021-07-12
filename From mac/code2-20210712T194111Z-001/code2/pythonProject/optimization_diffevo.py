from scipy.optimize import differential_evolution
from tqdm import tqdm
import numpy as np
from modulation import drr_norm_measure2
from collections import defaultdict
from collections import OrderedDict

ratio_min, ratio_max = 2, 10
omega1_min, omega1_max = 1, 10
retardance1_min, retardance1_max = 90, 180
retardance2_min, retardance2_max = 90, 180
determination_min, determination_max = 16, 100
t1_min, t1_max = 0, 360
t2_min, t2_max = 0, 360

# [determination_min, determination_max],
bounds = [[retardance1_min, retardance1_max],
          [retardance2_min, retardance2_max],

          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],

          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max],
          [t1_min, t1_max]]

pt1 = np.array([115.24248836, 178.06051382, 355.78534121, 168.59990806, 340.80028789,
               61.30995745, 236.71705076,  96.6036598,   50.74152919, 250.81706647,
               135.05244705, 234.02259508,  47.89636963, 199.01002683, 188.97590281,
               18.06415739, 331.94661235,  87.43800671,  62.04025622, 114.66593022,
               253.68441766, 229.84741657, 166.99826457, 283.25716879, 126.1542939,
               281.95617166,  57.61458249,  76.99885336,  73.89217446, 192.72278392,
               142.45711186, 334.06528553,  55.76432368, 320.43993281])

pt = np.array([115.24542188, 178.05975153,  355.78011154, 168.61084134, 340.78660485,
               61.32837037,  236.72134659,  96.60396579,  50.73292409,  250.82078187,
               135.03474647, 234.00240233,  47.91289559,  199.01785265, 188.97021451,
               18.06428162,  331.94459908,  87.42911845,  62.00849339,  114.66837877,
               253.68207691, 229.84383907,  166.99791251, 283.25156219, 126.15906427,
               281.95792843, 57.61311997,   77.00414725,  73.90307496,  192.71948865,
               142.45563967, 334.07168733,  55.76321723,  320.44190587])


for i in range(0, 1):
    result = differential_evolution(drr_norm_measure2,
                                    bounds,
                                    maxiter=2000,
                                    disp=True)
    pt = result['x']
    solution = pt
    sa_results = defaultdict()
    sa_results_reformat = OrderedDict()
    evaluation = drr_norm_measure2(solution)
    sa_results[0] = [solution[2:18], solution[18:34], evaluation]

    for i in range(0, 16):
        sa_results_reformat[solution[2 + i]] = solution[2 + i + 16]

final_dictionary = OrderedDict(sorted(sa_results_reformat.items()))

with open('result_differential.csv', 'w') as f:
    for key in final_dictionary.keys():
        f.write("%s,%s\n" % (key, final_dictionary[key]))

print('Solution: f(%s) = %.5f' % (solution, evaluation))


# ratio_min, ratio_max = 2, 10
# omega1_min, omega1_max = 1, 10
# retardance1_min, retardance1_max = 90, 180
# retardance2_min, retardance2_max = 90, 180
# determination_min, determination_max = 16, 100
#
# bounds = [[ratio_min, ratio_max],
#           [omega1_min, omega1_max],
#           [retardance1_min, retardance1_max],
#           [retardance2_min, retardance2_max],
#           [determination_min, determination_max]]
#
# result = differential_evolution(drr_norm_measure,
#                                 bounds,
#                                 disp=True)  # initial_temp=5e4
#
# print('Status : %s' % result['message'])
# print('Total Evaluations: %d' % result['nfev'])
# solution = result['x']
# evaluation = drr_norm_measure(solution)
#
# for i in tqdm(range(0, 2)):
#
#     result = differential_evolution(drr_norm_measure,
#                                     bounds)
#     solution_new = result['x']
#     evaluation_new = drr_norm_measure(solution)
#     print('\nSolution: f(%s) = %.5f' % (solution_new, evaluation_new))
#
#     if evaluation_new < evaluation:
#         evaluation = evaluation_new
#         solution = solution_new
#
# solution_newprint('Solution: f(%s) = %.5f' % (solution, evaluation))