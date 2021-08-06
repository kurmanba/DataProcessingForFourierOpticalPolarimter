from plot_helper import *
from simulation_parameters import *
from fourier_transforms import *
from scipy.optimize import leastsq, basinhopping  # , differential_evolution
from muller_calculations import *
from tqdm import tqdm


t_experiment, sampling_rate, s_0, omega_1, omega_2 = extract_parameters()                       # Extract parameters
ratios = np.arange(1, 5, 1)
store_results = defaultdict()
store_results_noise = defaultdict()

for i in tqdm(ratios):
    store_results[i], t = run_simulation2(t_experiment, sampling_rate, s_0, omega_1, omega_1*i)


#for i in ratios:
 #   plt.plot(t, store_results[i] + 0.125*i)

# plt.show()

plot_ratios(t, 5, store_results)

x_initial_basin = np.ones(10)
harmonics = 1
func = lambda x: f_annealing(x, t, store_results[harmonics])

optimized = basinhopping(func,
                         x_initial_basin,
                         minimizer_kwargs={"method": "L-BFGS-B"},
                         niter=300,
                         disp=False)

leastsq_x = leastsq(f_residual,
                    x_initial_basin,
                    args=(t, store_results[harmonics]),
                    maxfev=1000)

print(optimized.x[9])
plot_mc_fits(optimized.x, leastsq_x[0], t, store_results, harmonics)


# z = []
# z1 = []
# z2 = []
#
# for i in tqdm(range(0, 30000)):
#
#     z.append(drr_norm_measure_padua(np.array([90, 90])))
#     z2.append(drr_norm_measure(np.array([3, 10, 90, 90, 21])))
#
# fontsize = 10
# plt.hist(z2, bins=300, label="Linear Increments with ratio")
# plt.hist(z, bins=300, label="Padua Interpolation Points")
# plt.xlim(0, 400)
# plt.legend(loc='upper right')
# plt.xlabel("Data", fontsize=fontsize)
# plt.ylabel("Occurrence", fontsize=fontsize)
# plt.show()