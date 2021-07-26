from numpy.linalg import inv
from matplotlib import pyplot as plt
from muller_calculations import *
from inerpolation_methods import *
from tqdm import tqdm


def rotation_matrix2(teta, sign) -> np.ndarray:

    a, b = np.cos(2 * teta * sign), np.sin(2 * teta * sign)

    rotate = np.array([[1, 0, 0, 0],
                       [0, a, b, 0],
                       [0, -b, a, 0],
                       [0, 0, 0, 1]], np.float64)
    return rotate


def modulation_matrix2(theta1: any,                                            # From publication J. Zallat et. all 2006
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
    angle1: rotation of the wave plate 1
    angle2: rotation of the wave plate 2
    retardance1:
    retardance2:

    Return:
    ______
    p: Polarization parameter matrix
    """

    t_1 = MullerOperators(theta1, retardance1, 'LP_0')
    # r_0, r_1 = rotation_matrix2(90, -1), rotation_matrix2(90, 1)
    # r_2, r_3 = rotation_matrix2(90, -1), rotation_matrix2(90, 1)

    w_1 = t_1.general_wave_plate()                               # Wave plate transfer matrix at specified angle
    p_1 = t_1.linear_polarizer()                                 # Linear polarizer transfer matrix at specified angle
    # p_1 = r_0 @ p_1 @ r_1
    t_2 = MullerOperators(theta2, retardance2, 'LP_90')
    w_2 = t_2.general_wave_plate()                               # Wave plate transfer matrix at specified angle
    p_2 = t_2.linear_polarizer()                                 # Linear polarizer transfer matrix at specified angle
    # p_2 = r_2 @ p_2 @ r_3

    # Stoke Vectors
    # s_center = np.array([1, np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)])
    # s_test = np.array([1, 1, 1, 1])
    s_in = np.array([1, 0, 0, 0])

    g = w_1 @ p_1 @ s_in
    a = p_2 @ w_2
    p = np.kron(g, a[0][:])

    return p


def map_performance(ratio: float,
                    angular_increments: int):

    theta_1 = np.linspace(0, 180, angular_increments)
    theta_2 = np.linspace(0, 180, angular_increments)

    X, Y = np.meshgrid(theta_1, theta_2)
    z = np.zeros((len(theta_1), len(theta_2)))

    determination = 120
    t1 = np.linspace(0, 300, determination)
    t2 = np.linspace(0, ratio*300, determination)

    for i, x in enumerate(tqdm(theta_1)):
        for j, y in enumerate(theta_2):
            h = modulation_matrix2(t1[0], t2[0], x, y)
            for q in range(1, determination):
                h2 = modulation_matrix2(t1[q], t2[q], x, y)
                h = np.vstack((h, h2))

            norm_upperbound = 200
            try:
                inverse = np.linalg.inv(h)
            except np.linalg.LinAlgError:
                # print("Not Invertible")
                d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf))
                if d < norm_upperbound:
                    z[i][j] = d
                else:
                    z[i][j] = norm_upperbound
            else:
                # print("Invertible")
                d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.inv(h), np.inf))
                if d < norm_upperbound:
                    z[i][j] = d
                else:
                    z[i][j] = norm_upperbound

    theta_1i = np.linspace(0, 360, 360)
    condition = []

    for i, x in (enumerate(theta_1i)):
        h = modulation_matrix2(t1[0], t2[0], x, x)
        for q in range(1, determination):
            h2 = modulation_matrix2(t1[q], t2[q], x, x)
            h = np.vstack((h, h2))
        cond = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf))

        if cond < 50:
            condition.append(cond)
        else:
            condition.append(200)

    plt.plot(theta_1i, condition)
    # plt.savefig("condsss_vs_retardance_symmetric{}_{}.jpeg".format(ratio, determination))
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, z, cmap='jet')
    c = fig.colorbar(cp)
    c.set_ticks([])
    # plt.contourf(X, Y, z, 5, alpha=0.75, cmap='jet')
    # plt.contour(X, Y, z, 3, colors='black', linewidth=0.5)
    ax.set_title('PSA vs PSG at modulation ratio = {}'.format(ratio))
    ax.set_xlabel('retardance of PSG [deg]')
    ax.set_ylabel('retardance of PSA [deg]')
    plt.savefig("ratiosss{}_{}.jpeg".format(ratio, determination), dpi=1000)
    plt.show()


def drr_norm_measure(v: np.ndarray) -> float:                                  # optimization of performance parameters

    ratio = v[0]
    omega1 = v[1]
    retardance1 = v[2]
    retardance2 = v[3]
    determination = int(v[4])

    t1 = np.linspace(0, determination*omega1, determination)
    t2 = np.linspace(0, ratio*determination*omega1, determination)

    noise1 = np.random.normal(0, 1, len(t1))
    noise2 = np.random.normal(0, 1, len(t1))

    t1 = t1 + noise1
    t2 = t2 + noise2

    h = modulation_matrix2(t1[0], t2[0], retardance1, retardance2)
    for q in range(1, determination):
        h = np.vstack((h, modulation_matrix2(t1[q], t2[q], retardance1, retardance2)))

    return np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf)


def drr_norm_measure2(v: np.ndarray) -> float:                                    # optimization of interpolation points

    retardance1 = v[0]
    retardance2 = v[1]
    t1 = v[2:18]
    t2 = v[18:34]

    noise1 = np.random.normal(0, 1, len(t1))
    noise2 = np.random.normal(0, 1, len(t1))

    # t1 = t1 + noise1
    # t2 = t2 + noise2

    h = modulation_matrix2(t1[0], t2[0], retardance1, retardance2)
    for q in range(1, len(t1)):
        h = np.vstack((h, modulation_matrix2(t1[q], t2[q], retardance1, retardance2)))

    return np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf)


def drr_norm_measure_padua(v: np.ndarray) -> float:                    # optimization of interpolation points with padua

    retardance1 = v[0]
    retardance2 = v[1]
    t1, t2 = padua_points_2(5)

    noise1 = np.random.normal(0, 1, len(t1))
    noise2 = np.random.normal(0, 1, len(t1))

    t1 = t1 + noise1
    t2 = t2 + noise2

    h = modulation_matrix2(t1[0], t2[0], retardance1, retardance2)
    for q in range(1, len(t1)):
        h = np.vstack((h, modulation_matrix2(t1[q], t2[q], retardance1, retardance2)))

    return np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf)



