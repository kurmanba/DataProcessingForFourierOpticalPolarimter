from collections import defaultdict
import numpy as np


class MullerOperators:
    """
    The class contains dictionary with Mueller matrix of passive
    optical elements and can imitate functionality of PSG and PSA.
    """

    def __init__(self, teta, sigma, pl_type):

        self.teta, self.sigma, self.pl_type = teta, sigma, pl_type

    def wave_plate(self):

        r_0, r_1 = self.rotation_matrix(-1), self.rotation_matrix(1)
        wave_plate = self.retardance_matrix()

        return r_0 @ wave_plate @ r_1

    def linear_polarizer(self):

        linear_polarizer = 0.5 * np.array(self.muller_matrix(self.pl_type))

        return linear_polarizer

    def rotation_matrix(self, sign) -> np.ndarray:

        a, b = np.cos(2 * self.teta * sign), np.sin(2 * self.teta * sign)

        rotate = np.array([[1, 0, 0, 0],
                           [0, a, b, 0],
                           [0, -b, a, 0],
                           [0, 0, 0, 1]], np.float64)
        return rotate

    def retardance_matrix(self) -> np.ndarray:

        a, b = np.cos(self.sigma), np.sin(self.sigma)

        retardance = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, a, b],
                               [0, 0, -b, a]], np.float64)
        return retardance

    @staticmethod
    def muller_matrix(operator: str) -> defaultdict:                               # Dictionary of passive elements

        muller = defaultdict()
        muller['LP_0'] = np.array([[1, 1, 0, 0],
                                   [1, 1, 0, 0],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0]], np.float64)

        muller['LP_90'] = np.array([[1, -1, 0, 0],
                                    [-1, 1, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0]], np.float64)

        # additional elements should be added here to be used in the simulation
        muller['M_x'] = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]], np.float64)
        return muller[operator]


def transfer_matrix(theta1: any,
                    theta2: any,
                    retardance1: any,
                    retardance2: any,
                    incident: np.ndarray) -> np.ndarray:    # Muller matrix of the sample must be added

    """
    This function uses muller operations from the
    class Muller_operators and imitates behavior of
    the experimental optical setup. Incident light
    is transformed through series of linear operators.

    Parameters:
    __________
    angle1: rotation of the wave pallet 1
    angle2: rotation of the wave pallet 2
    incident: stoke vector of light emitted by laser

    Return:
    ______
    transform: Intensity of the light at CCD
    """

    t_1 = MullerOperators(theta1, retardance1, 'LP_0')
    w_1 = t_1.wave_plate()                                      # Wave plate transfer matrix at specified angle
    p_1 = t_1.linear_polarizer()                                # Linear polarizer transfer matrix at specified angle
    t_2 = MullerOperators(theta2, retardance2, 'LP_90')
    w_2 = t_2.wave_plate()                                      # Wave plate transfer matrix at specified angle
    p_2 = t_2.linear_polarizer()                                # Linear polarizer transfer matrix at specified angle

    s_out = p_2 @ w_2 @ w_1 @ p_1 @ incident

    if s_out[0] != np.sqrt(s_out[1]**2 + s_out[2]**2 + s_out[3]**2):
        print("Stoke Vector is corrupted during transformation!")

    return s_out[0]


def generate_rotation(t: np.ndarray,
                      w: float,
                      inverse: bool):
    """
    Function calculates angle of incidence based on angular rotation
    Parameters:
    __________
    t: time
    w: angular speed degree/seconds

    Return:
    ______
    angle at time instance t
    """

    if not inverse:
        return w*t
    else:
        return 360 - w*t % 360


def run_simulation(t_experiment: float,
                   sampling_rate: float,
                   s_0: np.ndarray,
                   omega_1: float,
                   omega_2: float) -> tuple:                                    # Inputs are fixed for now

    t_array = np.arange(0, t_experiment, 1/(sampling_rate * t_experiment))
    ccd_s = []                                                                  # Signal sampled by camera
    theta1 = generate_rotation(t_array, omega_1, inverse=False)
    theta2 = generate_rotation(t_array, omega_2, inverse=False)

    for i, j in enumerate(t_array):                                              # This needs to be VECTORIZED later on

        ccd_s.append(transfer_matrix(theta1[i], theta2[i], 133, 71, s_0))        # This needs to be VECTORIZED later on

    return ccd_s, t_array


