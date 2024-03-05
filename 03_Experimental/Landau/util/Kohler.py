import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

from scipy.interpolate import PchipInterpolator
from util.data import resolve_monotone

q = 1.602E-19

def read_params(filename):
    with open(filename, 'r') as f:
        params = []
        for line in f.readlines()[1:]:
            params.append([float(x) for x in line.split('\t')])
    return np.array(params)

def plot_rho(Ts, Bs, rho_xxs, cutoff, resu_dir):
    base = 1000
    colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, ax = plt.subplots(1, 1, figsize = (8,7))
    for T, rho_xx in zip(np.flip(Ts, axis = 0), np.flip(rho_xxs, axis = 0)):
        ax.plot(Bs[cutoff:], rho_xx[cutoff:]*1E2, '-', color = colors[int(T)-2], linewidth = 3)

    ax.set_xlabel(r'$B$ [T]', fontsize = 30)
    ax.set_ylabel(r'$\rho_{xx}$ [$\Omega$ cm]', fontsize = 30)
    ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    ax.set_xlim((np.min(Bs[cutoff:]), np.max(Bs[cutoff:])))
    ax.yaxis.get_offset_text().set_size(26)

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)), ax=plt.gca(), )
    cb.ax.set_title(r'$T$ [K]', fontsize = 30)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 26, which = 'both', direction = 'out')
    cb.ax.set_yscale('log')
    cb.ax.set_yticks([10, 100])
    cb.ax.set_yticklabels([10, 100])

    f.tight_layout()

    f.savefig(os.path.join(resu_dir, 'rho_vs_B.png'))
    plt.close()


def plot_MR(Ts, Bs, MRs, cutoff, resu_dir):
    base = 1000
    colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, ax = plt.subplots(1, 1, figsize = (8,7))
    for T, MR in zip(np.flip(Ts, axis = 0), np.flip(MRs, axis = 0)):
        ax.plot(Bs[cutoff:], MR[cutoff:], '-', color = colors[int(T)-2], linewidth = 3)
    ax.set_xlabel(r'$B$ [T]', fontsize = 30)
    ax.set_ylabel(r'MR [%]', fontsize = 30)
    
    ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    ax.set_xlim((np.min(Bs[cutoff:]), np.max(Bs[cutoff:])))
    ax.set_yticks([0, 1e4])
    ax.yaxis.get_offset_text().set_size(26)

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)), ax=plt.gca())
    cb.ax.set_title(r'$T$ [K]', fontsize = 30)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 26, which = 'both', direction = 'out')
    cb.ax.set_yscale('log')
    cb.ax.set_yticks([10, 100])
    cb.ax.set_yticklabels([10, 100])

    f.tight_layout()

    f.savefig(os.path.join(resu_dir, 'MR_vs_B.png'))
    plt.close()


def plot_MRK(Ts, Bs, MRs, rho_xx0s, cutoff, resu_dir):
    base = 1000
    colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, ax = plt.subplots(1, 1, figsize = (8,7))
    for T, MR, rho_xx0 in zip(np.flip(Ts, axis = 0), np.flip(MRs, axis = 0), np.flip(rho_xx0s, axis = 0)):
        plt.plot(Bs[cutoff:]/rho_xx0*1E-2, MR[cutoff:], '-', color = colors[int(T)-2], linewidth = 3)
    ax.set_xlabel(r'$B/\rho_{0}$ [T $\Omega^{-1}$ cm$^{-1}$]', fontsize = 30)
    ax.set_ylabel(r'MR [%]', fontsize = 30)
    ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_size(26)

    ax.set_xscale('log')
    ax.set_yscale('log')

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)))
    cb.ax.set_title(r'$T$ [K]', fontsize = 30)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 26, which = 'both', direction = 'out')
    cb.ax.set_yscale('log')
    cb.ax.set_yticks([10, 100])
    cb.ax.set_yticklabels([10, 100])

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'MR_vs_B_rho0.png'))
    plt.close()


def MR_shift(xs0, ys0, xs, ys):
    ymin = max(np.min(ys0), np.min(ys))
    ymax = min(np.max(ys0), np.max(ys))
    mask = (ys < ymax) * (ys > ymin)
    mask0 = (ys0 < ymax) * (ys0 > ymin)
    x_interp = PchipInterpolator(*resolve_monotone(ys[mask > 0], xs[mask > 0]), extrapolate = False)
    return x_interp(ys0[mask0 > 0]) / xs0[mask0 > 0]


def plot_MREK(Ts, Bs, MRs, rho_xx0s, cutoff, resu_dir, params_file = None):
    base = 1000
    colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, ax = plt.subplots(1, 1, figsize = (8,7))
    nTs = []
    for T, MR, rho_xx0 in zip(np.flip(Ts, axis = 0), np.flip(MRs, axis = 0), np.flip(rho_xx0s, axis = 0)):
        rho0nTs = MR_shift(Bs[cutoff:], MRs[-1][cutoff:], Bs[cutoff:], MR[cutoff:])*rho_xx0s[-1]
        rho0nT = np.mean(rho0nTs[~np.isnan(rho0nTs)])
        nTs.append(rho0nT/rho_xx0)
        plt.plot(Bs[cutoff:]/rho0nT*1E-2, MR[cutoff:], '-', color = colors[int(T)-2], linewidth = 3)
    nTs = np.flip(np.array(nTs), axis = 0)
    ax.set_xlabel(r'$B/\rho_{0}n_T$ [T $\Omega^{-1}$ cm$^{-1}$]', fontsize = 30)
    ax.set_ylabel(r'MR [%]', fontsize = 30)
    ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set_size(26)

    ax.set_xscale('log')
    ax.set_yscale('log')

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)))
    cb.ax.set_title(r'$T$ [K]', fontsize = 30)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 26, which = 'both', direction = 'out')
    cb.ax.set_yscale('log')
    cb.ax.set_yticks([10, 100])
    cb.ax.set_yticklabels([10, 100])

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'MR_vs_B_rho0nT.png'))
    plt.close()

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    plt.plot(Ts, nTs, '-o', color = '#E33119', linewidth = 3,  markersize=10, label = 'Kohler')
    axs.set_xlabel(r'$T$ [K]', fontsize = 30)
    axs.set_ylabel(r'$n_T$ [norm. at 300 K]', fontsize = 30)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
    axs.yaxis.get_offset_text().set_size(26)

    plt.legend(fontsize = 30)

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'nT_vs_T_ExtendedKohler.png'))
    plt.close()

    if params_file:
        # from two-band model
        params = read_params(params_file)
        ts = params[:, 0]
        uhs = params[:, 1]
        nhs = params[:, 3]
        ues = params[:, 5]
        nes = params[:, 7]
        nts = q * (uhs * nhs + ues * nes) ** (3/2) / (uhs ** 3 * nhs + ues ** 3 *nes) ** (1/2)

        f, ax = plt.subplots(1, 1, figsize = (8,7))
        ax.plot(ts, nts*1E-6, '-o', color = '#2F349A', linewidth = 3,  markersize=10, label = 'Two Band')
        ax.set_xlabel(r'$T$ [K]', fontsize = 30)
        ax.set_ylabel(r'$n_T$ [C cm$^{-3}$]', fontsize = 30)
        ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
        ax.yaxis.get_offset_text().set_size(26)
        plt.legend(fontsize = 30)

        f.tight_layout()
        f.savefig(os.path.join(resu_dir, 'nT_vs_T_TwoBand.png'))
        plt.close()

        f, axs = plt.subplots(1, 1, figsize = (8,7))
        plt.plot(Ts, nTs, '-o', color = '#E33119', linewidth = 3,  markersize=10, label = 'Kohler')
        plt.plot(ts, nts/nts[-1], '-o', color = '#2F349A', linewidth = 3,  markersize=10, label = 'Two Band')
        axs.set_xlabel(r'$T$ [K]', fontsize = 30)
        axs.set_ylabel(r'$n_T$ [norm. at 300 K]', fontsize = 30)
        axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
        axs.yaxis.get_offset_text().set_size(26)
        plt.legend(fontsize = 30, loc = 'upper left')

        f.tight_layout()
        f.savefig(os.path.join(resu_dir, 'nT_vs_T_compare.png'))
        plt.close()