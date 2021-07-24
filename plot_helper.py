from matplotlib import pyplot as plt
from matplotlib import ticker, get_backend, rc
import numpy as np
from fourier_transforms import *
from collections import defaultdict
from scipy.signal import find_peaks
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# This is utility file! Wrapper containing the style and characteristics of a plots are written.
# No description or comments as this is just a supplementary part of the code.

grey, gold, lightblue, green = '#808080', '#cab18c', '#0096d6', '#008367'
pink, yellow, orange, purple = '#ef7b9d', '#fbd349', '#ffa500', '#a35cff'
darkblue, brown, red = '#004065', '#731d1d', '#E31937'

_int_backends = ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg',
                 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo',
                 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo']

_backend = get_backend()

if _backend in _int_backends:
    fontsize = 12
    fig_scale = 1
else:
    fontsize = 12
    fig_scale = 1

quiver_params = {'angles': 'xy',
                 'scale_units': 'xy',
                 'scale': 1,
                 'width': 0.012}

grid_params = {'linewidth': 0.1,
               'alpha': 0.1}


def set_rc(func):
    def wrapper(*args, **kwargs):
        rc('font', family='serif', size=fontsize)
        rc('figure', dpi=500)
        rc('axes', axisbelow=True, titlesize=5)
        rc('lines', linewidth=0.5)
        func(*args, **kwargs)
    return wrapper


def set_rc2(func):
    def wrapper(*args, **kwargs):
        rc('font', family='serif', size=fontsize)
        # rc('figure', dpi=500)
        rc('axes', axisbelow=True, titlesize=12)
        rc('lines', linewidth=1.5)
        func(*args, **kwargs)
    return wrapper


@set_rc
def plot_initial_signal_double(x: np.ndarray,
                               y: np.ndarray) -> None:

    figure, (ax0, ax1) = plt.subplots(2, 1)
    ax0.grid(True, **grid_params)
    ax0.plot(x, y)
    ax1.scatter(x, y, color='#CD2305', s=18, marker='o')
    ax0.set_xlim([min(x) * 1.0, max(x) * 1.0])
    ax0.set_ylim([-max(y) * 0.2, max(y) * 1.2])
    # axis.set_aspect('equal')
    plt.show()


@set_rc
def plot_initial_signal(x: np.ndarray,
                        y: np.ndarray) -> None:

    fig, ax = plt.subplots()
    ax.grid(True, **grid_params)

    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    dx = xticks[1] - xticks[0]
    dy = yticks[1] - yticks[0]
    base = max(int(min(dx, dy)), 1)  # grid interval is always an integer
    loc = ticker.MultipleLocator(base=base)
    # ax.xaxis.set_major_locator(loc)

    ax.grid(True, **grid_params)

    markerline, stemline, baseline, = ax.stem(x, y, linefmt='#CD2305', markerfmt='o', basefmt='k.')
    plt.setp(stemline, linewidth=.5)
    plt.setp(markerline, markersize=.5)
    ax.set_xlim([min(x) * 1.0, max(x) * 1.0])
    ax.set_ylim([-max(y) * 0.2, max(y) * 1.2])
    ax.set_title('Data from the POM', fontsize=fontsize)
    plt.xlabel("Time [s]")
    plt.ylabel("Intensity Raw")
    plt.savefig("measurement.jpeg")
    plt.show()


@set_rc
def plot_shannon_single(x: np.ndarray,
                        y: np.ndarray,
                        x_0: np.ndarray,
                        y_0: np.ndarray) -> None:

    peaks, _ = find_peaks(y, height=40)
    fig, ax = plt.subplots()
    ax.grid(True, **grid_params)

    # ax.spines['left'].set_position('center')
    # ax.spines['bottom'].set_position('center')
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    dx = xticks[1] - xticks[0]
    dy = yticks[1] - yticks[0]
    base = max(int(min(dx, dy)), 1)  # grid interval is always an integer
    loc = ticker.MultipleLocator(base=base)
    # ax.xaxis.set_major_locator(loc)

    ax.grid(True, **grid_params)
    markerline, stemline, baseline, = ax.stem(x, y, linefmt='#CD2305', markerfmt='o', basefmt='k.')
    plt.setp(stemline, linewidth=.5)
    plt.setp(markerline, markersize=.5)
    # plt.plot(x, y, color='#CD2305', linewidth=0.9)
    # plt.scatter(x_0, y_0, s=3, marker='o')
    # arrow for periods
    arrowx = np.linspace(x[peaks[0]], x[peaks[1]], 100)
    arrowy = np.ones(len(arrowx))*49

    plt.scatter(x[peaks[0:2]], y[peaks[0:2]], s=7, marker='p', color=darkblue)
    plt.plot(arrowx, arrowy, color=darkblue, linewidth=.5)
    ax.text(4.6, 50.65, 'T1', color='darkblue', fontsize=fontsize)

    arrowx1 = np.linspace(x[peaks[0]+8], x[peaks[1]+8], 100)
    arrowy1 = np.ones(len(arrowx1))*16

    plt.scatter(x[peaks[0:2]+8], y[peaks[0:2]+8], s=7, marker='p', color=darkblue)
    plt.plot(arrowx1, arrowy1, color=darkblue, linewidth=.5)
    ax.text(4.6, 18, 'T2', color='darkblue', fontsize=fontsize)

    ax.set_xlim([min(x) * 1.0, max(x) * 1.0])
    ax.set_ylim([-max(y) * 0.2, max(y) * 1.2])
    ax.set_title('Whittaker–Shannon interpolation', fontsize=fontsize)
    plt.xlabel("Time [s]")
    plt.ylabel("Intensity Raw")
    plt.savefig("shannon.jpeg")
    plt.show()


@set_rc
def plot_detected_intensity(x: np.ndarray,
                            y: np.ndarray) -> None:

    fig, ax = plt.subplots()
    ax.spines['left'].set_linewidth(0.3)
    ax.spines['bottom'].set_linewidth(0.3)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.grid(True, **grid_params)
    plt.plot(x, y, color=darkblue, linewidth=2)
    ax.text(4.6, 18, 'T2', color='darkblue', fontsize=fontsize)

    ax.set_xlim([min(x) * 1.0, max(x) * 1.0])
    ax.set_ylim([-max(y) * 0.2, max(y) * 1.2])
    ax.set_title('w1/w2 = 5', fontsize=fontsize)
    plt.xlabel("Time [s]")
    plt.ylabel("Intensity Raw")

    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter('{x:.0f}')
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(0.5))

    plt.savefig("2.jpeg")
    plt.show()


@set_rc2
def plot_ratios(t: np.ndarray,
                max_ratio: int,
                results: defaultdict):

    for i in range(1, max_ratio):

        ax = plt.subplot((max_ratio - 1)//2, (max_ratio - 1)//2, i)
        ax.spines['left'].set_linewidth(0.3)
        ax.spines['bottom'].set_linewidth(0.3)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.title('Angular Speed Ratio: {}'.format(i), size=fontsize)
        plt.plot(t, results[i])
        plt.ylabel("Intensity Raw")
        plt.tight_layout(pad=.5)
        plt.grid(False)

    # plt.stem(X)
    plt.xlabel("Time [s]")
    plt.savefig("ratios1.jpeg")
    plt.show()


@set_rc2
def plot_mc_fits(variables_vector1: np.ndarray,
                 variables_vector2: np.ndarray,
                 t: np.ndarray,
                 results: defaultdict,
                 harmonics: int) -> None:

    plt.plot(t, f_series2(variables_vector1, t), label="Fourier Series Heuristics (Basinhopping)")
    plt.plot(t, f_series2(variables_vector2, t), label="Fourier Series Levenberg-Marquardt")
    plt.scatter(t, results[harmonics], label=" Data")
    plt.xlabel("t")
    plt.ylabel("Intensity")
    plt.legend(loc='upper right')
    plt.show()