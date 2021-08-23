import matplotlib.pyplot as plt

from modulation import *


def condition_number(theta1: np.ndarray,                                          # optimization of interpolation points
                     theta2: np.ndarray) -> float:

    h = modulation_matrix2(theta1[0], theta2[0], 90, 90)
    for q in range(1, len(theta1)):
        h = np.vstack((h, modulation_matrix2(theta1[q], theta2[q], 90, 90)))

    return np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf)


def generate_linear_sampling_angles(sampling_points: int,
                                    sampling_ratio: int) -> tuple:

    theta1 = np.linspace(0, 360, sampling_points)
    theta2 = sampling_ratio * theta1 % 360
    omega1 = 360/len(theta1)
    omega2 = sampling_ratio * omega1

    return theta1, theta2, omega1, omega2


z = np.arange(1, 10, 1)
j = np.arange(5, 13, 1)
points = np.arange(16, 91, 1)
l = []
for i in points:
    x, y, omega1, omega2 = generate_linear_sampling_angles(i, 20)
    l.append(condition_number(x, y))
    # print(condition_number(x, y))

for i in j:
    x, y = padua_points_2(i)
    print(condition_number(x, y), len(x))

plt.plot(l)
plt.show()
# t1, t2 = padua_points_2(8)
#
# cond_linear = condition_number(x, y)
# cond_padua = condition_number(t1, t2)
#
# print(cond_linear, cond_padua)
#
# print(len(t1), len(t2))
#
# plt.scatter(x, y)
# plt.scatter(t1, t2)
# plt.show()



