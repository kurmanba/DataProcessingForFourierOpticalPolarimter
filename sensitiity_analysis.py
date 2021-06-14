from fourier_transforms import fourier_coeffs, f_series, LagrangePoly    # helper functions
from fourier_transforms import f_series2, f_residual, f_annealing
from inerpolation_methods import whittaker_shannon_interpolation
from scipy.signal import find_peaks
from scipy.optimize import leastsq, basinhopping
from collections import deque
import numpy as np
from matplotlib import pyplot as plt                                      # graphing utilities
from plot_helper import plot_initial_signal_single,\
    plot_initial_signal_double, plot_shannon_single, set_rc
import pyswarms as ps                                                     # optimizers
from pyswarms.utils.functions import single_obj as fx
import pandas as pd                                                       # data acquisition
from tqdm import tqdm


df = pd.read_csv('camera_data.csv', index_col=None)
df = pd.DataFrame(df).to_numpy()
x, y = df[:, 0], df[:, 1]
window = [3, 13]                                                          # Specify range of data to be analyzed
clear = np.where(np.logical_and((x >= window[0]), (x <= window[1])))
x_0, y_0 = x[clear], y[clear]


s_freq = (max(x) - min(x)) / (len(x) - 1)
print("Sampling frequency is {} Hz".format(1/s_freq))

x_m_shanon = np.linspace(min(x_0), max(x_0), 1000)
y_m_shanon = (whittaker_shannon_interpolation(x_m_shanon, x_0, y_0))
# plot_shannon_single(x_m_shanon, y_m_shanon, x_0, y_0)
shift = -10
peaks, _ = find_peaks(y_m_shanon, height=40)
y_m_new, x_m_new = y_m_shanon[peaks[1] + shift:peaks[4] + shift], x_m_shanon[peaks[1] + shift:peaks[4] + shift]
plot_shannon_single(x_m_new, y_m_new, x_m_shanon[peaks[1] + shift:peaks[4] + shift], y_m_shanon[peaks[1] + shift:peaks[4] + shift])

# y_m1, x_m1 = y_0[peaks[2]:peaks[3]], x_0[peaks[2]:peaks[3]]
# y_mp, x_mp = y_0[peaks[1]:peaks[5]], x_0[peaks[1]:peaks[5]]

#
# a_0s = []
# a_0bs = []
# for shift in tqdm(range(-1, 0)):
#     # shift = -1
#     #
#     y_m_new, x_m_new = y_m_shanon[peaks[3] + shift:peaks[4] + shift], x_m_shanon[peaks[3] + shift:peaks[4] + shift]
#     x_initial = np.ones(25)*2
#     func = lambda x: f_annealing(x, x_m_new, y_m_new)
#     minimizer_kwargs = {"method": "BFGS"}
#     res = basinhopping(func, x_initial, minimizer_kwargs=minimizer_kwargs, niter=118)
#     x_initial_bh = res.x
#     leastsq_x = leastsq(f_residual, x_initial, args=(x_m_new, y_m_new), maxfev=10000)
#
#     # print(res.x)
#     a_0s.append(leastsq_x[0])
#     a_0bs.append(res.x)
#     print(np.array(a_0s) - np.array(a_0bs))

# print(res)
# plt.plot(a_0s)
# plt.plot(a_0bs)
# plt.show()


# print(leastsq_x[0][0])
# print("scipy.optimize.leastsq: ", leastsq_x[0])
# plt.plot(x_m_new, y_m_new, label="Data")
# @set_rc
# font = {'family': 'serif',
#         'color':  'darkred',
#         'weight': 'normal',
#         'size': 5,
#         }
# plt.plot(x_m_new, f_series2(leastsq_x[0], x_m_new), label="Fourier Series Levenberg-Marquardt", fontdict=font)
# plt.plot(x_m_new, f_series2(res.x, x_m_new), label="Fourier Series Heuristics (Simulated Annealing)", fontdict=font)
# plt.scatter(x_m_new, y_m_new, s=2, marker='o', label="Resampled Data (Whittaker-Shannon)", fontdict=font)
# plt.xlabel("t")
# plt.ylabel("Intensity")
# plt.legend(loc='upper right')
# plt.show()


# peak_dist = [x_0[peaks[i]] - x_0[peaks[i - 1]] for i in range(1, len(peaks))]
# period = x_0[peaks[2]] - x_0[peaks[1]]  # np.mean(peak_dist)


# plot_initial_signal_single(x_f, f_series(a_0, a_k, b_k, x_f, 4, period))



# x_f = np.linspace(min(x_m), max(x_m), len(x_m))
# y_f = lp.interpolate(x_f)
# print(len(a_k))

# a_0, a_k, b_k = fourier_coeffs(y_m_new)
# x_fourier = np.linspace(min(x_m_new), max(x_m_new), len(x_m_new))
# plt.plot(x_fourier, f_series(a_0, a_k, b_k, x_fourier, 4, period))
# plt.scatter(x_0, y_0)
# plt.show()


# plt.plot(x_m_new, y_m_new)
# plt.scatter(x_0, y_0)
# plt.show()


