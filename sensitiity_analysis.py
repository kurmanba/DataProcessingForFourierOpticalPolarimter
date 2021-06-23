from fourier_transforms import *                                           # helper functions
from inerpolation_methods import whittaker_shannon_interpolation
from scipy.signal import find_peaks
from scipy.optimize import leastsq, basinhopping, differential_evolution
import numpy as np
from matplotlib import pyplot as plt                                       # graphing utilities
from plot_helper import *
import pandas as pd                                                        # data acquisition
from tqdm import tqdm


df = pd.read_csv('camera_data.csv', index_col=None)
df = pd.DataFrame(df).to_numpy()
x, y = df[:, 0], df[:, 1]
window = [3, 13]                                                          # Specify range of data to be analyzed
clear = np.where(np.logical_and((x >= window[0]), (x <= window[1])))
x_0, y_0 = x[clear], y[clear]


s_freq = (max(x) - min(x)) / (len(x) - 1)
print("Sampling frequency is {} Hz".format(1/s_freq))

x_m_shanon = np.linspace(min(x_0), max(x_0), 2000)
y_m_shanon = (whittaker_shannon_interpolation(x_m_shanon, x_0, y_0))
# plot_shannon_single(x_m_shanon, y_m_shanon, x_0, y_0)

shift = -0
peaks, _ = find_peaks(y_m_shanon, height=40)
y_m_new, x_m_new = y_m_shanon[peaks[1] + shift:peaks[4] + shift], x_m_shanon[peaks[1] + shift:peaks[4] + shift]
# plot_shannon_single(x_m_new, y_m_new,
# x_m_shanon[peaks[1] + shift:peaks[4] + shift], y_m_shanon[peaks[1] + shift:peaks[4] + shift])

# y_m1, x_m1 = y_0[peaks[2]:peaks[3]], x_0[peaks[2]:peaks[3]]
# y_mp, x_mp = y_0[peaks[1]:peaks[5]], x_0[peaks[1]:peaks[5]]


a_0s = []
a_0bs = []
a_1bs = []
a_2bs = []
freqs = []

for shift in tqdm(range(0, 200, 10)):

    y_m_new, x_m_new = y_m_shanon[peaks[1] + shift:peaks[2] + shift], x_m_shanon[peaks[1] + shift:peaks[2] + shift]
    x_initial = np.ones(10) * 2

    func = lambda x: f_annealing(x, x_m_new, y_m_new)
    func2 = lambda: f_sgd(x, x_m_new, y_m_new)

    minimizer_kwargs = {"method": "L-BFGS-B"}
    basinhop = basinhopping(func, x_initial, minimizer_kwargs=minimizer_kwargs, niter=10)
    # leastsq_x = leastsq(f_residual, x_initial, args=(x_m_new, y_m_new), maxfev=2000)
    print(basinhop.fun)
    # bounds = ([-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100],
    #           [-100, 100], [-100, 100], [-100, 100], [-100, 100])
    # diff_evo = differential_evolution(func, bounds)

    # cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
    # var = tf.Variable(x_initial)
    #
    # for counter in tqdm(range(100)):
    #     sgd.minimize(func2,  var_list=[var])

    # a_0s.append(leastsq_x[0])
    a_0bs.append(basinhop.x[0])
    a_1bs.append(basinhop.x[1])
    a_2bs.append(basinhop.x[2])
    freqs.append(basinhop.x[-1])
    # a_0bs.append(diff_evo.x)


# print(diff_evo.x)
# plt.plot(a_0s)
print(freqs)
plt.plot(a_0bs)
plt.plot(a_1bs)
plt.plot(a_2bs)
plt.show()


# print(leastsq_x[0][0])
# print("scipy.optimize.leastsq: ", leastsq_x[0])
# plt.plot(x_m_new, y_m_new, label="Data")
# @set_rc

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 5,
        }

# plt.plot(x_m_new, f_series2(leastsq_x[0], x_m_new), label="Fourier Series Levenberg-Marquardt")
# plt.plot(x_m_new, f_series2(basinhop.x, x_m_new), label="Fourier Series Heuristics (Simulated Annealing)")
# #plt.plot(x_m_new, f_series2(var.numpy, x_m_new), label="Fourier Series Heuristics (SGD)")
#
# plt.scatter(x_m_new, y_m_new, s=2, marker='o', label="Resampled Data (Whittaker-Shannon)")
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


