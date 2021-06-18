from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


class MullerOperators:

    """
    The class was written to imitate elements of optical setup.
    See function transfer matrix on how to use class.
    """

    def __init__(self, teta, sigma):

        self.teta = teta
        self.sigma = sigma
        self.sign = 1
        self.sign2 = -1

    def wave_pallet(self):

        r_0 = self.rotation_matrix1()
        r_1 = self.rotation_matrix2()
        matrix = self.retardance_matrix()

        return r_0 @ matrix @ r_1

    def linear_polarizer(self):

        r_0 = self.rotation_matrix1()
        r_1 = self.rotation_matrix2()
        matrix = self.muller_matrix('LP')
        return (r_0 @ matrix @ r_1)/2

    def rotation_matrix1(self) -> np.ndarray:

        a = np.cos(2 * self.teta * self.sign)
        b = np.sin(2 * self.teta * self.sign)

        rotate = np.array([[1, 0, 0, 0],
                           [1, a, b, 0],
                           [0, -b, a, 0],
                           [0, 0, 0, 1]], np.float64)
        return rotate

    def rotation_matrix2(self) -> np.ndarray:

        a = np.cos(2 * self.teta * self.sign2)
        b = np.sin(2 * self.teta * self.sign2)

        rotate = np.array([[1, 0, 0, 0],
                           [1, a, b, 0],
                           [0, -b, a, 0],
                           [0, 0, 0, 1]], np.float64)
        return rotate

    def retardance_matrix(self) -> np.ndarray:

        a = np.cos(2 * self.sigma)
        b = np.sin(2 * self.sigma)

        retardance = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, a, b],
                               [0, 0, -b, a]], np.float64)
        return retardance

    @staticmethod
    def muller_matrix(operator: str) -> defaultdict:

        muller = defaultdict()
        muller['LP'] = np.array([[1, 1, 0, 0],
                                 [1, 1, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], np.float64)

        muller['wp'] = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 0, 0],
                                 [0, 0, 0, 0]], np.float64)

        return muller[operator]


def transfer_matrix(angle1: any,
                    angle2: any,
                    incident: np.ndarray) -> np.ndarray:    # Muller matrix of the sample could be added
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
    transform: Stoke vector of the light at detector
    """
    t_1 = MullerOperators(angle1, 0)
    w_1 = t_1.wave_pallet()                         # wave pallet transfer matrix at specified angle
    p_1 = t_1.linear_polarizer()                    # linear polarizer transfer matrix at specified angle
    t_2 = MullerOperators(angle2, 0)
    w_2 = t_2.wave_pallet()                         # wave pallet transfer matrix at specified angle
    p_2 = t_2.linear_polarizer()                    # linear polarizer transfer matrix at specified angle

    transform = p_2 @ w_2 @ w_1 @ p_1 @ incident

    return transform


def generate_rotation(t: np.ndarray,
                      w: float):
    """
    Function calculates angle of incidence based on angular rotation
    Parameters:
    __________
    t: time
    w: angular speed

    Return:
    ______
    angle at time instance t
    """
    return w*t


z = np.zeros((360, 360))
time = np.arange(0, 3, 0.001)
omega_1 = 30
omega_2 = 10
theta1 = generate_rotation(time, omega_1)
theta2 = generate_rotation(time, omega_2)

# theta1 = np.arange(0, 360, 1)
# theta2 = np.arange(0, 360, 1)

s_0 = [1, 0, 0, 0]                               # initial stoke vector
s_0 = np.transpose(s_0)
b = []


# for i in tqdm(theta1):
#     for j in theta2:
#         intensity = transfer_matrix(i, j, s_0)
#         b.append(intensity[0])
for i, j in enumerate(time):
    l = transfer_matrix(theta1[i], theta2[i], s_0)[0]
    b.append(l)

plt.plot(time, b)
plt.show()
# z = np.reshape(b, (360, 360))
# xx, yy = np.meshgrid(theta1, theta2, sparse=True)
# h = plt.contourf(theta1, theta2, z)
# plt.show()
