from matplotlib import pyplot as plt
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
from collections import deque
from plot_helper import plot_initial_signal


df = pd.read_csv('camera_data.csv', index_col=None)
df = pd.DataFrame(df).to_numpy()

x, y = df[:, 0], df[:, 1]
window = [3, 13]

clear = np.where(np.logical_and((x >= window[0]), (x <= window[1])))
x_0, y_0 = x[clear], y[clear]

peaks, _ = find_peaks(y_0, height=40)
y_m = np.zeros(len(y_0))
y_m[peaks] = y_0[peaks]


peak_dist = [x_0[peaks[i]] - x_0[peaks[i - 1]] for i in range(1, len(peaks))]
data_points = np.mean(peak_dist) * 30

print(np.mean(peak_dist), data_points)
pattern_extracted = deque()

i = 0
while len(pattern_extracted) < data_points + 1:
    pattern_extracted.append(np.mean(y_0[peaks[1:-1]+i] + y_0[peaks[1:-1] - i])/2)
    i += 1

x_average = np.linspace(0, 10, 10*24)
y_n = np.zeros(len(x_average))
y_n[0::30] = [x+1 for x in y_n[0::30]]
peaks_1, _ = find_peaks(y_n, height=0.1)

for i in range(0, len(pattern_extracted)//2 + 1):
    y_n[peaks_1 + i] = pattern_extracted[i]
    y_n[peaks_1 - i] = pattern_extracted[i]

for i in range(0, len(pattern_extracted)//2):
    y_m[peaks + i] = pattern_extracted[i]
    y_m[peaks - i] = pattern_extracted[i]


duration = 10
sample_rate = 24
period = 1 / sample_rate

normalized = np.int16((y_m / y_m.max()) * 32767)
y_f = rfft(y_m)
x_f = rfftfreq(len(x_0), period)

plt.plot(x_f[1:], np.abs(y_f)[1:])
plt.show()

# FILTER
filter_noise, _ = find_peaks(y_f, height=100)
new_filter = np.where((y_f > 0))
print(type(filter_noise))
print(new_filter)
y_recon = np.zeros(len(y_f))
y_recon[new_filter] = y_f[new_filter]

print(filter_noise)
plt.plot(x_f[1:], np.abs(y_recon)[1:])
plt.show()

new_sig = irfft(y_recon)
norm_new_sig = np.int16(new_sig * new_sig.max() / 32767)

plot_initial_signal(x_0, y_m)

plt.plot(x_0[1:], new_sig)
plt.show()