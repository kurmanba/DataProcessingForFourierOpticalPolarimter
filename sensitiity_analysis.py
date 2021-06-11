from fourier_transforms import fourier_coeffs, f_series, LagrangePoly    # helper functions
from inerpolation_methods import whittaker_shannon_interpolation
from scipy.signal import find_peaks
from collections import deque
import numpy as np
from matplotlib import pyplot as plt                                      # graphing utilities
from plot_helper import plot_initial_signal_single,\
    plot_initial_signal_double, plot_shannon_single
import pyswarms as ps                                                     # optimizers
from pyswarms.utils.functions import single_obj as fx
import pandas as pd                                                       # data acquisition


df = pd.read_csv('camera_data.csv', index_col=None)
df = pd.DataFrame(df).to_numpy()
x, y = df[:, 0], df[:, 1]
window = [3, 13]                                                          # Specify range of data to be analyzed
clear = np.where(np.logical_and((x >= window[0]), (x <= window[1])))
x_0, y_0 = x[clear], y[clear]


s_freq = (max(x) - min(x)) / (len(x) - 1)
print("Sampling frequency is {} Hz".format(1/s_freq))


peaks, _ = find_peaks(y_0, height=40)

y_m, x_m = y_0[peaks[1]:peaks[2]], x_0[peaks[1]:peaks[2]]
y_m1, x_m1 = y_0[peaks[2]:peaks[3]], x_0[peaks[2]:peaks[3]]
y_mp, x_mp = y_0[peaks[1]:peaks[5]], x_0[peaks[1]:peaks[5]]

# plot_initial_signal_single(x_m, y_m)
peak_dist = [x_0[peaks[i]] - x_0[peaks[i - 1]] for i in range(1, len(peaks))]
period = x_0[peaks[2]] - x_0[peaks[1]]  # np.mean(peak_dist)


# plot_initial_signal_single(x_f, f_series(a_0, a_k, b_k, x_f, 4, period))

x_m_new = np.linspace(min(x_0), max(x_0), 1000)
y_m_new = (whittaker_shannon_interpolation(x_m_new, x_0, y_0, period))
plot_shannon_single(x_m_new, y_m_new, x_0, y_0)

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


