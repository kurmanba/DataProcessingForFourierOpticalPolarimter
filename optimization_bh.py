import numpy as np
from modulation import drr_norm_measure2
from scipy.optimize import basinhopping
from numpy.random import rand
from collections import defaultdict
from collections import OrderedDict

ratio_min, ratio_max = 2, 10
omega1_min, omega1_max = 1, 10
retardance1_min, retardance1_max = 90, 180
retardance2_min, retardance2_max = 90, 180
determination_min, determination_max = 16, 100
t1_min, t1_max = 0, 360


# Starting sample
pt = np.zeros(34)

pt[0] = retardance1_min + rand(1) * (retardance1_max - retardance1_min)
pt[1] = retardance2_min + rand(1) * (retardance2_max - retardance2_min)

pt[2] = t1_min + rand(1) * (t1_max - t1_min)
pt[3] = t1_min + rand(1) * (t1_max - t1_min)
pt[4] = t1_min + rand(1) * (t1_max - t1_min)
pt[5] = t1_min + rand(1) * (t1_max - t1_min)
pt[6] = t1_min + rand(1) * (t1_max - t1_min)
pt[7] = t1_min + rand(1) * (t1_max - t1_min)
pt[8] = t1_min + rand(1) * (t1_max - t1_min)
pt[9] = t1_min + rand(1) * (t1_max - t1_min)
pt[10] = t1_min + rand(1) * (t1_max - t1_min)
pt[11] = t1_min + rand(1) * (t1_max - t1_min)
pt[12] = t1_min + rand(1) * (t1_max - t1_min)
pt[13] = t1_min + rand(1) * (t1_max - t1_min)
pt[14] = t1_min + rand(1) * (t1_max - t1_min)
pt[15] = t1_min + rand(1) * (t1_max - t1_min)
pt[16] = t1_min + rand(1) * (t1_max - t1_min)
pt[17] = t1_min + rand(1) * (t1_max - t1_min)


pt[18] = t1_min + rand(1) * (t1_max - t1_min)
pt[19] = t1_min + rand(1) * (t1_max - t1_min)
pt[20] = t1_min + rand(1) * (t1_max - t1_min)
pt[21] = t1_min + rand(1) * (t1_max - t1_min)
pt[22] = t1_min + rand(1) * (t1_max - t1_min)
pt[23] = t1_min + rand(1) * (t1_max - t1_min)
pt[24] = t1_min + rand(1) * (t1_max - t1_min)
pt[25] = t1_min + rand(1) * (t1_max - t1_min)
pt[26] = t1_min + rand(1) * (t1_max - t1_min)
pt[27] = t1_min + rand(1) * (t1_max - t1_min)
pt[28] = t1_min + rand(1) * (t1_max - t1_min)
pt[29] = t1_min + rand(1) * (t1_max - t1_min)
pt[30] = t1_min + rand(1) * (t1_max - t1_min)
pt[31] = t1_min + rand(1) * (t1_max - t1_min)
pt[32] = t1_min + rand(1) * (t1_max - t1_min)
pt[33] = t1_min + rand(1) * (t1_max - t1_min)

pt1 = np.array([115.24248836, 178.06051382, 355.78534121, 168.59990806, 340.80028789,
               61.30995745, 236.71705076,  96.6036598,   50.74152919, 250.81706647,
               135.05244705, 234.02259508,  47.89636963, 199.01002683, 188.97590281,
               18.06415739, 331.94661235,  87.43800671,  62.04025622, 114.66593022,
               253.68441766, 229.84741657, 166.99826457, 283.25716879, 126.1542939,
               281.95617166,  57.61458249,  76.99885336,  73.89217446, 192.72278392,
               142.45711186, 334.06528553,  55.76432368, 320.43993281])

pt2 = np.array([115.24542188, 178.05975153,  355.78011154, 168.61084134, 340.78660485,
               61.32837037,  236.72134659,  96.60396579,  50.73292409,  250.82078187,
               135.03474647, 234.00240233,  47.91289559,  199.01785265, 188.97021451,
               18.06428162,  331.94459908,  87.42911845,  62.00849339,  114.66837877,
               253.68207691, 229.84383907,  166.99791251, 283.25156219, 126.15906427,
               281.95792843, 57.61311997,   77.00414725,  73.90307496,  192.71948865,
               142.45563967, 334.07168733,  55.76321723,  320.44190587])

pt = np.array([115.24550269, 178.06048972,  355.77904242, 168.61190208, 340.78717667,
               61.33041125,   236.72085122,  96.60397408,  50.73599883,  250.81962195,
               135.04623846,  234.00983873,  47.9239721,   199.02098451, 188.96516411,
               18.06415407,   331.94451021,  87.42728847,  62.01569289,  114.66853441,
               253.68192381,  229.84157718,  166.99835798, 283.25269748, 126.15630983,
               281.95795649,  57.61422293,   76.99911241,  73.91106751,  192.71492852,
               142.44877113,  334.07026418,  55.76233125,  320.4425484])

pt5 = np.array([115, 178,  355, 169, 341,
               61,   237,  97,  51,  251,
               135,  234,  48,  199, 189,
               18,   332,  87,  62,  115,
               253,  230,  167, 283, 126,
               281,  58,   78,  74,  193,
               142,  334,  56,  320])

print(drr_norm_measure2(pt))
step_size = np.array([1/10, 1/100, 1/1000])
iteration_size = np.array([50, 50, 50])

for i, j in enumerate(step_size):
    result = basinhopping(drr_norm_measure2,
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
sa_results[0] = [solution[2:18], solution[18:34], evaluation]

for i in range(0, 16):
    sa_results_reformat[solution[2 + i]] = solution[2 + i + 16]

final_dictionary = OrderedDict(sorted(sa_results_reformat.items()))

with open('result_basinhopping.csv', 'w') as f:
    for key in final_dictionary.keys():
        f.write("%s,%s\n" % (key, final_dictionary[key]))

print('Solution: f(%s) = %.5f' % (solution, evaluation))
