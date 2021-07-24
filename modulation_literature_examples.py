import numpy as np
from numpy.linalg import inv
from muller_calculations import MullerOperators


def modulation_matrix(theta1: any,
                      theta2: any,
                      retardance1: any,
                      retardance2: any) -> float:
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
    w_1 = t_1.general_wave_plate()                                      # Wave plate transfer matrix at specified angle
    p_1 = t_1.linear_polarizer()                                # Linear polarizer transfer matrix at specified angle
    t_2 = MullerOperators(theta2, retardance2, 'LP_90')
    w_2 = t_2.general_wave_plate()                                      # Wave plate transfer matrix at specified angle
    p_2 = t_2.linear_polarizer()                                # Linear polarizer transfer matrix at specified angle
    c = np.zeros((4, 4))

    for k in range(0, 4):
        for l in range(0, 4):
            for j in range(0, 4):
                c[k][l] += p_1[0][j]*w_1[j][k]*w_2[l][j]*p_2[j][0]

    d = c

    try:
        inverse = np.linalg.inv(d)
    except np.linalg.LinAlgError:
        # print("Not Invertible")
        return -1000
    else:
        # print("Invertible")
        return np.linalg.norm(d, np.inf) * np.linalg.norm(np.linalg.inv(d), np.inf)


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
    angle1: rotation of the wave plate 1
    angle2: rotation of the wave plate 2
    retardance1:
    retardance2:

    Return:
    ______
    p: Polarization parameter matrix
    """

    t_1 = MullerOperators(theta1, retardance1, 'LP_+45')
    w_1 = t_1.general_wave_plate()  # Wave plate transfer matrix at specified angle
    p_1 = t_1.linear_polarizer()  # Linear polarizer transfer matrix at specified angle
    t_2 = MullerOperators(theta2, retardance2, 'LP_-45')
    w_2 = t_2.general_wave_plate()  # Wave plate transfer matrix at specified angle
    p_2 = t_2.linear_polarizer()  # Linear polarizer transfer matrix at specified angle

    s_center = np.array([1, np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)])
    s_in = np.array([1, 0, 0, 0])
    s_test = np.array([1, 1, 1, 1])
    #
    g = w_1 @ p_1 @ s_in
    a = s_in @ p_2 @ w_2
    p = np.kron(np.transpose(g), a)

    return p
    # try:
    #     inverse = np.linalg.inv(p)
    # except np.linalg.LinAlgError:
    #     print("Not Invertible")
    #     return -100
    # else:
    #     # print("Invertible")
    #     return np.linalg.norm(p, 'fro') * np.linalg.norm(np.linalg.inv(p), 'fro')


def modulation_matrix3(theta1: any,                                 # From publication
                       theta2: any,
                       retardance1: any,
                       retardance2: any) -> float:
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

    c = np.zeros((4, 4))
# ______________________________________________________________________________________________________________________

    c[0][0] = 1
    c[0][1] = np.cos(2*theta1)**2 + np.cos(retardance1)*np.sin(2*theta1)**2
    c[0][2] = np.cos(2*theta1)*np.sin(2*theta1) - np.cos(retardance1)*np.cos(2*theta1)*np.sin(2*retardance1)
    c[0][3] = np.sin(retardance1)*np.sin(2*theta1)

# ______________________________________________________________________________________________________________________

    c[1][0] = -(np.cos(2*theta2)**2 - np.cos(retardance2)*np.sin(2*theta2)**2)
    c[1][1] = -(np.cos(2*theta2)**2 - np.cos(retardance2)*np.sin(2*theta2)**2) * \
               (np.cos(2*theta1)**2 + np.cos(retardance1)*np.sin(2*theta1)**2)
    c[1][2] = -(np.cos(2*theta2)**2 - np.cos(retardance2)*np.sin(2*theta2)**2) * \
               (np.cos(2*theta1)*np.sin(2*theta1) - np.cos(retardance1)*np.cos(2*theta1)*np.sin(2*retardance1))
    c[1][3] = -(np.sin(retardance1)*np.sin(2*theta1)) * \
               (np.cos(2*theta2)**2 - np.cos(retardance2)*np.sin(2*theta2)**2)

# ______________________________________________________________________________________________________________________

    c[2][0] = np.cos(retardance2)*np.cos(2*theta2)*np.sin(2*theta2) - np.cos(2*theta2)*np.sin(2*theta2)
    c[2][1] = -(np.cos(2*theta1)**2 + np.cos(retardance1)*np.sin(2*theta1)**2) *\
               (np.cos(2*theta2)*np.sin(2*theta2) - np.cos(retardance2)*np.cos(2*theta2)*np.sin(2*theta2))
    c[2][2] = -(np.cos(2*theta2)*np.sin(2*theta2) - np.cos(retardance2)*np.cos(2*theta2)*np.sin(2*theta2)) * \
               (np.cos(2*theta1)*np.sin(2*theta1) - np.cos(retardance2)*np.cos(2*theta1)*np.sin(2*theta1))
    c[2][3] = -(np.sin(retardance1)*np.sin(2*theta1)) * \
               (np.cos(2*theta2)*np.sin(2*theta2) - np.cos(retardance2)*np.cos(2*theta2)*np.sin(2*retardance2))

# ______________________________________________________________________________________________________________________

    c[3][0] = np.sin(retardance2)*np.sin(2*theta2)
    c[3][1] = np.sin(retardance2)*np.sin(2*theta2)*(np.cos(2*theta1)**2 + np.cos(retardance1)*np.sin(2*theta1)**2)
    c[3][2] = (np.sin(retardance2)*np.sin(2*theta2)) * \
              (np.cos(2*theta1)*np.sin(2*theta1) - np.cos(retardance1)*np.cos(2*theta1)*np.sin(2*theta1))
    c[3][3] = np.sin(retardance1)*np.sin(2*theta1)*np.sin(retardance2)*np.sin(2*theta2)

# ______________________________________________________________________________________________________________________
    c_t = np.transpose(c)
    g = c @ c_t
    # g_1 = inv(g)
    # r = c_t @ g_1
    # np.linalg.det(c_t @ c)
    # sum(np.square(np.diagonal(c_t @ c)))
    d = c
    try:
        inverse = np.linalg.inv(d)
    except np.linalg.LinAlgError:
        # print("Not Invertible")
        return -100
    else:
        # print("Invertible")
        return np.linalg.norm(d, np.inf) * np.linalg.norm(np.linalg.inv(d), np.inf)


def modulation_matrix4(theta1: any,                             # From the book Handbook of Optics
                       theta2: any,
                       retardance1: any,
                       retardance2: any,
                       q: any) -> np.ndarray:
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
    w: Polarization parameter matrix
    """

    w = np.zeros(16)
    dt = theta1-theta2
    c1 = np.cos(retardance1/2)
    c2 = np.cos(retardance2/2)
    s1 = np.sin(retardance1/2)
    s2 = np.sin(retardance2/2)
# ______________________________________________________________________________________________________________________

    w[0] = 1
    w[1] = c1**2 + s1**2 * np.cos(4*q*dt)
    w[2] = s1**2 * np.sin(4*q*dt)
    w[3] = np.sin(retardance1)*np.sin(2*q*dt)

# ______________________________________________________________________________________________________________________

    w[4] = c2**2 + s2**2*np.cos(20*q*dt)
    w[5] = c1**2 * c2**2 + s1**2 * c2**2 * np.cos(4*q*dt) + c1**2 * s2**2 * np.cos(20*dt*1) + \
        0.5*s1**2*s2**2*(np.cos(16*q*dt)+np.cos(24*q*dt))
    w[6] = s1**2 * c2**2 * np.sin(4*q*dt) + 0.5*s1**2 * s2**2 * (-np.sin(16*q*dt) + np.sin(24*q*dt))
    w[7] = np.sin(retardance1)*c2**2 * np.sin(2*q*dt) + 0.5*np.sin(retardance1)*c2**2 * \
        (-np.sin(18*q*dt)+np.sin(22*q*dt))

# ______________________________________________________________________________________________________________________

    w[8] = s2**2*np.sin(20*q*dt)
    w[9] = c1**2*s2**2 * np.sin(20*q*dt) + 0.5*s1**2 * s2**2 * (np.sin(16*q*dt) + np.sin(24*q*dt))
    w[10] = 0.5*s1**2*s2**2*(np.cos(16*q*dt) - np.cos(24*q*dt))
    w[11] = 0.5*s1*s2**2*(np.cos(18*q*dt) - np.cos(22*q*dt))

# ______________________________________________________________________________________________________________________

    w[12] = -np.sin(retardance2)*np.sin(10*q*dt)
    w[13] = -c1**2*np.sin(retardance2)*np.sin(10*q*dt) - 0.5*s1**2*np.sin(retardance2)*(np.sin(6*q*dt) +
                                                                                        np.sin(14*q*dt))
    w[14] = -0.5*s1**2*np.sin(retardance2)*(np.cos(6*q*dt)-np.cos(14*q*dt))
    w[15] = -0.5*np.sin(retardance1)*np.sin(retardance2)*(np.cos(8*q*dt)-np.cos(12*q*dt))

# ______________________________________________________________________________________________________________________
    return np.array(w)
