from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy.linalg import inv
from scipy import linalg
from muller_calculations import *
from random import seed
from random import choice
from random import sample


def modulation_matrix2(theta1: any,                                     # From publication J. Zallat et. all 2006
                       theta2: any,
                       retardance1: any,
                       retardance2: any) -> np.ndarray:
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
    w_1 = t_1.general_wave_plate()                               # Wave plate transfer matrix at specified angle
    p_1 = t_1.linear_polarizer()                                 # Linear polarizer transfer matrix at specified angle
    t_2 = MullerOperators(theta2, retardance2, 'LP_90')
    w_2 = t_2.general_wave_plate()                               # Wave plate transfer matrix at specified angle
    p_2 = t_2.linear_polarizer()                                 # Linear polarizer transfer matrix at specified angle

    s_center = np.array([1, np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)])
    s_in = np.array([1, 0, 0, 0])
    s_test = np.array([1, 1, 1, 1])

    g = w_1 @ p_1 @ np.array([1, 0, 0, 0])
    a = s_in @ p_2 @ w_2
    p = np.kron(a, g)

    return p


theta_1 = np.linspace(0, 90, 10)
theta_2 = np.linspace(0, 90, 10)

X, Y = np.meshgrid(theta_1, theta_2)
z = np.zeros((len(theta_1), len(theta_2)))

sequence1 = list(np.linspace(0, 360, 500))
sequence2 = list(np.linspace(0, 360, 500))

determination = 120
t_array = np.arange(0, 1, 1/determination)
t1 = generate_rotation(t_array, 360, inverse=False)
t2 = generate_rotation(t_array, 360*3, inverse=False)
print(t1)

# t1 = sample(sequence1, determination)
# t2 = sample(sequence2, determination)

for i, x in (enumerate(tqdm(theta_1))):
    for j, y in enumerate(theta_2):
        h = modulation_matrix2(t1[0], t2[0], x, y)
        for q in range(1, determination):
            h2 = modulation_matrix2(t1[q], t2[q], x, y)
            h = np.vstack((h, h2))
        try:
            inverse = np.linalg.inv(h)
        except np.linalg.LinAlgError:
            # print("Not Invertible")
            d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf))
            if d < 50:
                z[i][j] = d
            else:
                z[i][j] = 50
        else:
            # print("Invertible")
            d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.inv(h), np.inf))
            if d < 50:
                z[i][j] = d
            else:
                z[i][j] = 50


theta_1i = np.linspace(0, 360, 360)
condition = []

for i, x in (enumerate(tqdm(theta_1i))):
    h = modulation_matrix2(t1[0], t2[0], x, x)
    for q in range(1, determination):
        h2 = modulation_matrix2(t1[q], t2[q], x, x)
        h = np.vstack((h, h2))
    cond = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf))

    if cond < 50:
        condition.append(cond)
    else:
        condition.append(100)


plt.plot(theta_1i, condition)
plt.show()

fig, ax = plt.subplots(1, 1)
cp = ax.contourf(X, Y, z)
fig.colorbar(cp)
ax.set_title('PSA vs PSG (retardance)')
ax.set_xlabel('PSG')
ax.set_ylabel('PSA')
plt.show()
