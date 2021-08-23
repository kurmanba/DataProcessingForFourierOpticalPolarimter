import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def get_w_matrix(theta1: float,
                 theta2: float,
                 i: int) -> float:
    """
    The following function returns transfer matrix of the DRR with quarter wave plates.

    Parameter:
    _________
    theta1: azimuthal angle of wave plate 1
    theta2: azimuthal angle of wave plate 2

    Return:
    ______
    w: Transfer matrix of DRR
    """

    w = np.zeros(16)
    w[0] = np.abs(1/4)
    w[1] = np.abs(-1/8*np.cos(4*theta2) - 1/8)
    w[2] = np.abs(-1/8*np.sin(4*theta2))
    w[3] = np.abs(-1/4*np.sin(2*theta2))
    w[4] = np.abs(1/8*np.cos(4*theta1) + 1/8)
    w[5] = np.abs(-1/16*np.cos(4*theta1) - 1/32*np.cos(4*theta1 + 4*theta2) - 1/32*np.cos(-4*theta1 + 4*theta2) -
                  1/16*np.cos(4*theta2) - 1/16)
    w[6] = np.abs(-1/32*np.sin(4*theta1 + 4*theta2) - 1/32*np.sin(-4*theta1 + 4*theta2) - 1/16*np.sin(4*theta2))
    w[7] = np.abs(-1/16*np.sin(4*theta1 + 2*theta2) - 1/16*np.sin(-4*theta1 + 2*theta2) - 1/8*np.sin(2*theta2))
    w[8] = np.abs(1/8*np.sin(4*theta1))
    w[9] = np.abs(-1/16*np.sin(4*theta1) - 1/32*np.sin(4*theta1 + 4*theta2) + 1/32*np.sin(-4*theta1 + 4*theta2))
    w[10] = np.abs(1/32*np.cos(4*theta1 + 4*theta2) - 1/32*np.cos(-4*theta1 + 4*theta2))
    w[11] = np.abs(1/16*np.cos(4*theta1 + 2*theta2) - 1/16*np.cos(-4*theta1 + 2*theta2))
    w[12] = np.abs(-1/4*np.sin(2*theta1))
    w[13] = np.abs(1/8*np.sin(2*theta1) + 1/16*np.sin(2*theta1 + 4*theta2) - 1/16*np.sin(-2*theta1 + 4*theta2))
    w[14] = np.abs(-1/16*np.cos(2*theta1 + 4*theta2) + 1/16*np.cos(-2*theta1 + 4*theta2))
    w[15] = np.abs(-1/8*np.cos(2*theta1 + 2*theta2) + 1/8*np.cos(-2*theta1 + 2*theta2))
    # return -(w[i] - (sum(w) - w[i]))

    return w[i]



def get_min_max_expression(theta1: float,
                           theta2: float) -> float:
    i = 2
    w = get_w_matrix(theta1, theta2)
    return -(w[i] - (sum(w) - w[i]))


def map_mins(ind):

    # theta_1 = np.linspace(0, 3, 200)
    # theta_2 = np.linspace(0, 3, 200)

    theta_1 = np.linspace(0, 90, 100)
    theta_2 = np.linspace(0, 90, 100)

    z = np.zeros((len(theta_1), len(theta_2)))

    for i, x in enumerate(tqdm(theta_1)):
        for j, y in enumerate(theta_2):
            z[i][j] = get_w_matrix(x, y, ind)

    return z


theta_1 = np.linspace(0, 90, 100)
theta_2 = np.linspace(0, 90, 100)
X, Y = np.meshgrid(theta_1, theta_2)
fig, ax = plt.subplots(4, 4)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = ax.ravel()

for i in range(0, 16):
    z = map_mins(i)
    cp = ax[i].contourf(X, Y, z, cmap='jet')
    # c = fig.colorbar(cp)
# plt.contourf(X, Y, z, 5, alpha=0.75, cmap='jet')
# plt.contour(X, Y, z, 3, colors='black', linewidth=0.5)
    ax[i].set_title(r'$W[{}]$'.format(i))
    ax[i].set_xlabel(r'$\theta_1$  [deg]')
    ax[i].set_ylabel(r'$\theta_2$ [deg]')

plt.show()

# print(get_w_matrix(131.94689145, 14.92226221))

parser = argparse.ArgumentParser(description='lloyd-max iteration quantizer')
parser.add_argument('--bit', '-b', type=int, default=6, help='number of bit for quantization')
parser.add_argument('--iteration', '-i', type=int, default=30, help='number of iteration')

# parser.add_argument('--range', '-r', type=int, default=10, help='range of the initial distribution')
# parser.add_argument('--resolution', '-re', type=int, default=100, help='resolution of the initial distribution')
# parser.add_argument('--save_location', '-s', type=str, default='outputs', help='save location of representations and')

args = parser.parse_args()

