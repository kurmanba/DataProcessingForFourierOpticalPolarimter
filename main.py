import matplotlib.pyplot as plt
from plot_helper import *
from simulation_parameters import *
from collections import defaultdict
from fourier_transforms import *
from scipy.optimize import leastsq, basinhopping  # , differential_evolution
from tqdm import tqdm
import numpy as np


t_experiment, sampling_rate, s_0, omega_1, omega_2 = extract_parameters()                       # Extract parameters
ratios = np.arange(1, 5, 1)
store_results = defaultdict()
store_results_noise = defaultdict()

for i in tqdm(ratios):
    store_results[i], t = run_simulation(t_experiment, sampling_rate, s_0, omega_1, omega_1*i)


for i in ratios:
    plt.plot(t, store_results[i] + 0.5*i)

plt.show()

plot_ratios(t, 5, store_results)

x_initial_basin = np.ones(10)
harmonics = 1
func = lambda x: f_annealing(x, t, store_results[harmonics])

optimized = basinhopping(func,
                         x_initial_basin,
                         minimizer_kwargs={"method": "L-BFGS-B"},
                         niter=3000,
                         disp=False)

leastsq_x = leastsq(f_residual,
                    x_initial_basin,
                    args=(t, store_results[harmonics]),
                    maxfev=1000)

print(optimized.x[9])
plot_mc_fits(optimized.x, leastsq_x[0], t, store_results, harmonics)


