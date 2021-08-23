import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker, get_backend, rc


grey, gold, lightblue, green = '#808080', '#cab18c', '#0096d6', '#008367'
pink, yellow, orange, purple = '#ef7b9d', '#fbd349', '#ffa500', '#a35cff'
darkblue, brown, red = '#004065', '#731d1d', '#E31937'

g = np.load("sample.npy")                                                                          # Load CCD recordings

# Initialize arrays for storing statistical information
z, k, p, q = np.zeros((100, 600)), np.zeros((100, 600)), np.zeros((100, 600)), np.zeros((100, 600))
t_decay = 5300

y = g[34][113][:t_decay]
x = np.linspace(1, t_decay, t_decay)

y2 = np.sort(y)
y3 = y2

for i, j in enumerate(y2):
    y2[i] = round(y2[i] / 10) * 10

vbreaks = []
hbreaks = np.unique(y2)

for i in range(len(y2) - 1):
    if y2[i+1] > y2[i]:
        vbreaks.append(i)


quiver_params = {'angles': 'xy',
                 'scale_units': 'xy',
                 'scale': 0.1,
                 'width': 0.012}

grid_params = {'linewidth': 0.1,
               'alpha': 0.1}

fontsize = 8


def set_rc(func):
    def wrapper(*args, **kwargs):
        rc('font', family='sans-serif', size=20)
        rc('figure', dpi=700)
        rc('axes', axisbelow=True, titlesize=20)
        rc('lines', linewidth=0.1)
        rc('legend', fontsize=20, loc='upper left')
        func(*args, **kwargs)
    return wrapper


@set_rc
def plot_time_series(x, y):

    fig, ax1 = plt.subplots(1, figsize=(10, 5))
    ax1.plot(y, color=(183/255, 207/255, 246/255), linewidth=1, zorder=1)
    ax1.scatter(x=x, y=y, marker='o', edgecolor='k', color=(65/255, 105/255, 225/255), linewidth=0.2, s=20, zorder=2)
    ax1.set_title('Data from flow stop measurement')
    # ax1.axes.xaxis.set_ticklabels([])
    # ax1.axes.yaxis.set_ticklabels([])
    ax1.set_ylabel('$Intensity$ [raw]')
    ax1.set_xlabel('$N$ measurements')
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.85, wspace=None, hspace=None)
    plt.savefig("flow_stop_time_series.jpeg")
    # plt.show()


@set_rc
def plot_quantization(x, y2, y3):

    fig, ax1 = plt.subplots(1, figsize=(10, 5))
    # ax1.plot(y2, color=(183/255, 207/255, 246/255), linewidth=0.2, zorder=2)
    ax1.plot(y3, color=red, linewidth=0.5, zorder=2, label='CCD')

    ax1.scatter(x=x, y=y2, marker='o', edgecolor='k', color=(65/255, 105/255, 225/255),
                linewidth=0.01, s=2.5, zorder=3, label='Quantized')

    for line in vbreaks:
        ax1.vlines(line, ymin=1900, ymax=2600, color=(86/255, 101/255, 105/255),
                   linestyles='dashed', linewidth=1, zorder=1)

    for line in hbreaks:
        ax1.hlines(line, xmin=0, xmax=5300, color=(86/255, 101/255, 105/255),
                   linestyles='dashed', linewidth=1, zorder=1)

    ax1.set_title('Quantization')
    ax1.set_ylabel('$I$ [raw intensity]')
    ax1.set_xlabel('$N$ measurements')
    plt.subplots_adjust(left=0.15, bottom=0.2, right=0.95, top=0.85, wspace=None, hspace=None)
    plt.legend()
    plt.savefig("flow_stop_quantization.jpeg")
    # plt.show()


plot_time_series(x, y)
plot_quantization(x, y2, y3)
