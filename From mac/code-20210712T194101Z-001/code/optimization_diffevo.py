from modulation import drr_norm_measure
from scipy.optimize import differential_evolution
from tqdm import tqdm


ratio_min, ratio_max = 2, 10
omega1_min, omega1_max = 1, 10
retardance1_min, retardance1_max = 90, 180
retardance2_min, retardance2_max = 90, 180
determination_min, determination_max = 16, 100

bounds = [[ratio_min, ratio_max],
          [omega1_min, omega1_max],
          [retardance1_min, retardance1_max],
          [retardance2_min, retardance2_max],
          [determination_min, determination_max]]

result = differential_evolution(drr_norm_measure,
                                bounds,
                                disp=True)  # initial_temp=5e4

print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
solution = result['x']
evaluation = drr_norm_measure(solution)

for i in tqdm(range(0, 2)):

    result = differential_evolution(drr_norm_measure,
                                    bounds)
    solution_new = result['x']
    evaluation_new = drr_norm_measure(solution)
    print('\nSolution: f(%s) = %.5f' % (solution_new, evaluation_new))

    if evaluation_new < evaluation:
        evaluation = evaluation_new
        solution = solution_new

print('Solution: f(%s) = %.5f' % (solution, evaluation))