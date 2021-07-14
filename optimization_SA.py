from modulation import drr_norm_measure2
from scipy.optimize import dual_annealing
from tqdm import tqdm
import random
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

random.seed()
sa_results = defaultdict()
sa_results_reformat = defaultdict()

for i in tqdm(range(0, 2)):

    random.seed(i)
    result = dual_annealing(drr_norm_measure2,
                            bounds)
    solution = result['x']
    evaluation = drr_norm_measure2(solution)
    sa_results[i] = [solution[2:18], solution[18:34], evaluation]

    if evaluation < 15:
        for i in range(0, 16):
            sa_results_reformat[solution[2+i]] = solution[2+i+16]
    print('Solution: f(%s) = %.5f' % (solution, evaluation))

final_dictionary = OrderedDict(sorted(sa_results_reformat.items()))

with open('result_simulated_annealing.csv', 'w') as f:
    for key in final_dictionary.keys():
        f.write("%s,%s\n" % (key, final_dictionary[key]))

print(sa_results.values())
