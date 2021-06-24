import matplotlib.pyplot as plt
from simulation_parameters import *
from collections import defaultdict
from fourier_transforms import *
from scipy.optimize import leastsq, basinhopping  # , differential_evolution
from tqdm import tqdm
import numpy as np


t_experiment, sampling_rate, s_0, omega_1, omega_2 = extract_parameters()                       # Extract parameters
ratios = np.arange(1, 50, 1)
store_results = defaultdict()

for i in tqdm(ratios):
    store_results[i], t = run_simulation(t_experiment, sampling_rate, s_0, omega_1, omega_1*i)

for i in ratios:
    plt.plot(t, store_results[i] + 0.5*i)
plt.show()

x_initial_basin = np.ones(10)
harmonics = 2
func = lambda x: f_annealing(x, t, store_results[harmonics])

optimized = basinhopping(func,
                         x_initial_basin,
                         minimizer_kwargs={"method": "L-BFGS-B"},
                         niter=500,
                         disp=False)

leastsq_x = leastsq(f_residual,
                    x_initial_basin,
                    args=(t, store_results[harmonics]),
                    maxfev=1000)

print(optimized.x[9])
plt.plot(t, f_series2(optimized.x, t), label="Fourier Series Heuristics (Basinhopping)")
plt.plot(t, f_series2(leastsq_x[0], t), label="Fourier Series Levenberg-Marquardt")
plt.scatter(t, store_results[harmonics], label=" Data")
plt.xlabel("t")
plt.ylabel("Intensity")
plt.legend(loc='upper right')
plt.show()
