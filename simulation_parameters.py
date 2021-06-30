import numpy as np
from muller_calculations import *


def extract_parameters():

    # Initialization Simulation Parameters (All the parameters for the optical setup must be entered here)

    # 1. Time Variables
    # _________________
    t_experiment = .1                                                      # Duration of simulation [seconds]
    sampling_rate = 100000                                                  # CCD [frames/second] or 1/s_r [Hz]
    # 2. Incoming light properties
    # ____________________________
    s_0 = [1, 0, 0, 0]                                                    # Incident light in a stoke vector form
    # 3. Characteristics of optical setup (passive elements)
    # ______________________________________________________
    retardance_wp_1 = 133                                                 # Phase shift induced by optics (wave plate)
    retardance_wp_2 = 71                                                  # Phase shift induced by optics (wave plate)
    # 4. Characteristics of optical setup (active elements)
    # ______________________________________________________
    omega_1 = 360                                                         # Angular speed of wave plate [deg/second]
    omega_2 = 15                                                          # Angular speed of wave plate [deg/second]
    # 5. Additional Parameters to add later on
    # ______________________________________________________

    return t_experiment, sampling_rate, s_0, omega_1, omega_2
