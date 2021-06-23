from fourier_transforms import *     # helper functions
from fourier_transforms import f_series2, f_residual, f_annealing
from inerpolation_methods import *
from muller_calculations import *
from scipy.signal import find_peaks
from scipy.fftpack import fft, ifft, fftshift
from scipy.optimize import leastsq, basinhopping, differential_evolution
import numpy as np
from matplotlib import pyplot as plt                                      # graphing utilities
from plot_helper import *
import pandas as pd                                                       # data acquisition
from tqdm import tqdm


a_0s = []
a_0bs = []
a_1bs = []
a_2bs = []
freqs = []
t_experiment = 2
sampling_rate = 100  # [frames/second]
time = np.arange(0, t_experiment, 1 / (sampling_rate * t_experiment))

y_m_new, x_m_new = run_simulation()
X = fft(y_m_new[2], axis=0)

X = 2*np.abs(X) / (sampling_rate * t_experiment)
plt.plot(X)
plt.show()

for shift in tqdm(range(3, 4)):

    y_m_new = y_m_new[1]
    x_initial_basin = np.ones(10)*0
    func = lambda x: f_annealing(x, x_m_new, y_m_new)
    minimizer_kwargs = {"method": "L-BFGS-B"}
    basinhop = basinhopping(func, x_initial_basin, minimizer_kwargs=minimizer_kwargs, niter=500)
    x_initial_leven = np.ones(10)*20
    leastsq_x = leastsq(f_residual, x_initial_leven, args=(x_m_new, y_m_new), maxfev=1000)
    a_0bs.append(basinhop.x[0])
    a_1bs.append(basinhop.x[1])
    a_2bs.append(basinhop.x[2])
    freqs.append(basinhop.x[-1])
    # a_0bs.append(diff_evo.x)


print(freqs)
# plt.plot(a_0bs)
# plt.plot(a_1bs)
# plt.plot(a_2bs)
# plt.show()
a_0, a_k, b_k = fourier_coeffs(y_m_new)


plt.plot(x_m_new, f_series2(leastsq_x[0], x_m_new), label="Fourier Series Levenberg-Marquardt")
plt.plot(x_m_new, f_series2(basinhop.x, x_m_new), label="Fourier Series Heuristics (Simulated Annealing)")
# plt.plot(x_m_new, f_series2(var.numpy, x_m_new), label="Fourier Series Heuristics (SGD)")

# plt.scatter(x_m_new, y_m_new[3], s=2, marker='o', label="Resampled Data (Whittaker-Shannon)")
plt.plot(x_m_new, y_m_new, label=" Data")

plt.xlabel("t")
plt.ylabel("Intensity")
plt.legend(loc='upper right')
plt.show()
