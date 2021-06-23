import matplotlib.pyplot as plt
from simulation_parameters import *


t_experiment, sampling_rate, s_0, omega_1, omega_2 = extract_parameters()
ccd, t = run_simulation(t_experiment, sampling_rate, s_0, omega_1, omega_2)

plt.plot(t, ccd)
plt.show()
