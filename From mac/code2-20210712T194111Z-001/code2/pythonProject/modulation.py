import matplotlib.pyplot as plt
from numpy.linalg import inv
from muller_calculations import *
from padua import *
from tqdm import tqdm


def rotation_matrix2(teta: float,
                     sign: float) -> np.ndarray:

    a, b = np.cos(2 * teta * sign), np.sin(2 * teta * sign)

    rotate = np.array([[1, 0, 0, 0],
                       [0, a, b, 0],
                       [0, -b, a, 0],
                       [0, 0, 0, 1]], np.float64)
    return rotate


def modulation_matrix2(theta1: any,
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
    r_0, r_1 = rotation_matrix2(90, -1), rotation_matrix2(90, 1)
    r_2, r_3 = rotation_matrix2(90, -1), rotation_matrix2(90, 1)

    w_1 = t_1.general_wave_plate()                               # Wave plate transfer matrix at specified angle
    p_1 = t_1.linear_polarizer()                                 # Linear polarizer transfer matrix at specified angle
    # p_1 = r_0 @ p_1 @ r_1
    t_2 = MullerOperators(theta2, retardance2, 'LP_90')
    w_2 = t_2.general_wave_plate()                               # Wave plate transfer matrix at specified angle
    p_2 = t_2.linear_polarizer()                                 # Linear polarizer transfer matrix at specified angle
    # p_2 = r_2 @ p_2 @ r_3

    s_center = np.array([1, np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)])
    s_in = np.array([1, 0, 0, 0])
    s_test = np.array([1, 1, 1, 1])

    g = w_1 @ p_1 @ s_in
    a = p_2 @ w_2

    p = np.kron(g, a[0][:])

    return p


def map_performance(ratio: float,
                    omega: float,
                    angular_increments: int):

    determination = 83
    theta_1 = np.linspace(90, 170, angular_increments)
    theta_2 = np.linspace(90, 170, angular_increments)

    # theta_1 = np.linspace(13, 18.4, angular_increments)
    # theta_2 = np.linspace(13, 18.4, angular_increments)

    X, Y = np.meshgrid(theta_1, theta_2)
    z = np.zeros((len(theta_1), len(theta_2)))

    # t1 = np.linspace(0, determination*omega, determination)
    # t2 = np.linspace(0, ratio*determination*omega, determination)

    # t1 = [355.77904242, 168.61190208, 340.78717667,
    #       61.33041125, 236.72085122, 96.60397408, 50.73599883, 250.81962195,
    #       135.04623846, 234.00983873, 47.9239721, 199.02098451, 188.96516411,
    #       18.06415407, 331.94451021, 87.42728847]
    #
    # t2 = [62.01569289, 114.66853441,
    #       253.68192381, 229.84157718, 166.99835798, 283.25269748, 126.15630983,
    #       281.95795649, 57.61422293, 76.99911241, 73.91106751, 192.71492852,
    #       142.44877113, 334.07026418, 55.76233125, 320.4425484]

    t1, t2 = padua_points_2(8)
    t1 = t1
    t2 = t2
    # print(t1, "\n")
    # print(t2, "\n")

    # t1_space = np.linspace(0, 360, 1000)
    # t2_space = np.linspace(0, 360, 1000)
    #
    # t1 = random.sample(list(t1_space), determination)
    # t2 = random.sample(list(t2_space), determination)

    for i, x in enumerate(tqdm(theta_1)):
        for j, y in enumerate(theta_2):
            h = modulation_matrix2(t1[0], t2[0], x, y)
            for q in range(1, len(t1)):
                h = np.vstack((h, modulation_matrix2(t1[q], t2[q], x, y)))
            norm_upperbound = 50
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

    # for i, x in (enumerate(theta_1i)):
    #
    #     h = modulation_matrix2(t1[0], t2[0], x, x)
    #     for q in range(1, determination):
    #         h2 = modulation_matrix2(t1[q], t2[q], x, x)
    #         h = np.vstack((h, h2))
    #     cond = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf))
    #
    #     if cond < 50:
    #         condition.append(cond)
    #     else:
    #         condition.append(200)
    #
    # plt.plot(theta_1i, condition)
    # plt.savefig("condsss_vs_retardance_symmetric{}_{}.jpeg".format(ratio, determination))

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, z, cmap='jet')
    c = fig.colorbar(cp)
    # c.set_ticks([])
    # plt.contourf(X, Y, z, 5, alpha=0.75, cmap='jet')
    # plt.contour(X, Y, z, 3, colors='black', linewidth=0.5)
    ax.set_title('PSA vs PSG at modulation ratio = {}'.format(ratio))
    # ax.set_title('PSA vs PSG at stochastic rotation')
    ax.set_xlabel('retardance of PSG [deg]')
    ax.set_ylabel('retardance of PSA [deg]')
    plt.savefig("ratioshift{}_{}.jpeg".format(ratio, determination), dpi=1000)
    plt.show()


# ratios = np.array([3.68])
# #
# for i in ratios:
#     map_performance(i, 3.38026534, 100)


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

    # t1_space = np.linspace(0, 360, 1000)
    # t2_space = np.linspace(0, 360, 1000)
    #
    # t1 = random.sample(list(t1_space), determination)
    # t2 = random.sample(list(t2_space), determination)
    z = 0

    h = modulation_matrix2(t1[0], t2[0], retardance1, retardance2)
    for q in range(1, determination):
        h2 = modulation_matrix2(t1[q], t2[q], retardance1, retardance2)
        h = np.vstack((h, h2))

    norm_upperbound = 50000000

    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf))
        if d < norm_upperbound:
            z = d
        else:
            z = norm_upperbound
    else:
        d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.inv(h), np.inf))
        if d < norm_upperbound:
            z = d
        else:
            z = norm_upperbound
    return z


def drr_norm_measure2(v: np.ndarray) -> float:                                    # optimization of interpolation points

    retardance1 = v[0]
    retardance2 = v[1]
    t1 = v[2:18]
    t2 = v[18:34]

    noise1 = np.random.normal(0, 1, len(t1))
    noise2 = np.random.normal(0, 1, len(t1))

    t1 = t1 + noise1
    t2 = t2 + noise2

    h = modulation_matrix2(t1[0], t2[0], retardance1, retardance2)
    for q in range(1, len(t1)):
        h = np.vstack((h, modulation_matrix2(t1[q], t2[q], retardance1, retardance2)))

    norm_upperbound = 50000000

    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf))
        if d < norm_upperbound:
            z = d
        else:
            z = norm_upperbound
    else:
        d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.inv(h), np.inf))
        if d < norm_upperbound:
            z = d
        else:
            z = norm_upperbound
    return z


def drr_norm_measure_padua(v: np.ndarray) -> float:                    # optimization of interpolation points with padua

    retardance1 = v[0]
    retardance2 = v[1]
    t1, t2 = padua_points_2(5)

    # print(len(t1))

    noise1 = np.random.normal(0, 1, len(t1))
    noise2 = np.random.normal(0, 1, len(t1))
    t1 = t1 + noise1
    t2 = t2 + noise2

    h = modulation_matrix2(t1[0], t2[0], retardance1, retardance2)
    for q in range(1, len(t1)):
        h = np.vstack((h, modulation_matrix2(t1[q], t2[q], retardance1, retardance2)))

    norm_upperbound = 50000000

    try:
        inverse = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.pinv(h), np.inf))
        if d < norm_upperbound:
            z = d
        else:
            z = norm_upperbound
    else:
        d = (np.linalg.norm(h, np.inf) * np.linalg.norm(np.linalg.inv(h), np.inf))
        if d < norm_upperbound:
            z = d
        else:
            z = norm_upperbound
    return z


pt3 = np.array([90, 90,  355.77904242, 168.61190208, 340.78717667,
               61.33041125,   236.72085122,  96.60397408,  50.73599883,  250.81962195,
               135.04623846,  234.00983873,  47.9239721,   199.02098451, 188.96516411,
               18.06415407,   331.94451021,  87.42728847,  62.01569289,  114.66853441,
               253.68192381,  229.84157718,  166.99835798, 283.25269748, 126.15630983,
               281.95795649,  57.61422293,   76.99911241,  73.91106751,  192.71492852,
               142.44877113,  334.07026418,  55.76233125,  320.4425484])


pt4 = np.array([115.24550269, 178.06048972,  355.77904242, 168.61190208, 340.78717667,
               61.33041125,   236.72085122,  96.60397408,  50.73599883,  250.81962195,
               135.04623846,  234.00983873,  47.9239721,   199.02098451, 188.96516411,
               18.06415407,   331.94451021,  87.42728847,  62.01569289,  114.66853441,
               253.68192381,  229.84157718,  166.99835798, 283.25269748, 126.15630983,
               281.95795649,  57.61422293,   76.99911241,  73.91106751,  192.71492852,
               142.44877113,  334.07026418,  55.76233125,  320.4425484])


pt5 = np.array([115, 178,  355, 169, 341,
               61,   237,  97,  51,  251,
               135,  234,  48,  199, 189,
               18,   332,  87,  62,  115,
               253,  230,  167, 283, 126,
               281,  58,   78,  74,  193,
               142,  334,  56,  320])


pt6 = np.array([90, 90,  355, 169, 341,
               61,   237,  97,  51,  251,
               135,  234,  48,  199, 189,
               18,   332,  87,  62,  115,
               253,  230,  167, 283, 126,
               281,  58,   78,  74,  193,
               142,  334,  56,  320])

z = []
z1 = []
z2 = []

for i in tqdm(range(0, 30000)):

    z.append(drr_norm_measure_padua(np.array([90, 90])))
    z1.append(drr_norm_measure2(pt3))
    z2.append(drr_norm_measure(np.array([3, 10, 90, 90, 21])))

# plt.hist(z1, bins=100)
fontsize = 10
plt.hist(z2, bins=300, label="Linear Increments with ratio")
plt.hist(z, bins=300, label="Padua Interpolation Points")
plt.xlim(0, 400)
plt.legend(loc='upper right')
plt.xlabel("Data", fontsize=fontsize)
plt.ylabel("Occurrence", fontsize=fontsize)
plt.show()
