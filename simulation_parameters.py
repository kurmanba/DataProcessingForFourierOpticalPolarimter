import numpy as np
from muller_calculations import *


# Initialization Simulation Parameters
# 1. Time Variables
# _________________
t_experiment = 1                                                         # Duration of simulation [seconds]
sampling_rate = 1000                                                     # CCD [frames/second] or 1/s_r [Hz]
# 2. Incoming light properties
# ____________________________
s_0 = [1, 0, 0, 0]                                                       # Incident light in a stoke vector form
# 3. Characteristics of optical setup (passive elements)
# ______________________________________________________
retardance_wp_1 = 133                                                    # phase shift induced by optics (wave plate)
retardance_wp_2 = 71                                                     # phase shift induced by optics (wave plate)
# 4. Characteristics of optical setup (active elements)
# ______________________________________________________
omega_1 = 10                                                             # Angular speed of wave plate
omega_2 = 10                                                             # Angular speed of wave plate
# 5. Additional Parameters to add later on
# ______________________________________________________


def extract_simulation_parameters():

    return

# plot_ratios(time, 3, results)