import os
import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.interpolate import PchipInterpolator
from utils.data import resolve_monotone

# MR_S3
# sample's measurement data dir
data_file = 'D:/python_project/H-TaP/03_Experimental/TwoBand/S3_ch1_MR_ch2_Hall_@diffT.dat'

# thickness, width, and length of the sample
T = 0.4
W = 1.66
L = 0.56

# PPMS measurement channel number
ch = 1
########################################################


# field interpolation range and resolution
H_min = 0.0
H_max = 9.0
H_res = 1000

# extracted data file name
resu_name = data_file[:-4] + '_extracted.pkl'

# extract data
Bs = np.linspace(H_min, H_max, H_res)

rho_xxs = []
rho_xys = []

data = pd.read_csv(data_file)
data.rename(columns = {'Temperature (K)': 'Temperature', 
                        'Field (Oe)': 'Field', 
                        'Resistance Ch1 (Ohms)': 'rho_xx',
                        'Resistance Ch2 (Ohms)': 'rho_xy'},
            inplace = True)
    
data['Temperature'] = np.round(data['Temperature'], 0)  # round temperature to integer
data['Field'] = data['Field']/10000                     # convert field unit from Gauss to Tesla
data['rho_xx'] = data['rho_xx'] * W * T / (L * 1000)    # calculate sample resistivity
data['rho_xy'] = data['rho_xy'] * W * T / (L * 1000)    # calculate sample resistivity

Ts = np.sort(list(set(data['Temperature'])))

for T in Ts:
    data_T = data.loc[(data['Temperature'] == T)]

    rho_xx_0 = np.min(np.abs(data_T['rho_xx']))
    rho_xx_pos = data_T.loc[(data_T['Field'] > 0)].sort_values(by = 'Field', ascending = True)
    rho_xx_pos = rho_xx_pos[pd.to_numeric(rho_xx_pos['rho_xx'], errors='coerce').notnull()]
    rho_xx_neg = data_T.loc[(data_T['Field'] < 0)].sort_values(by = 'Field', ascending = False)
    rho_xx_neg = rho_xx_neg[pd.to_numeric(rho_xx_neg['rho_xx'], errors='coerce').notnull()]

    rho_xy_0 = np.min(np.abs(data_T['rho_xy']))
    rho_xy_pos = data_T.loc[(data_T['Field'] > 0)].sort_values(by = 'Field', ascending = True)
    rho_xy_pos = rho_xy_pos[pd.to_numeric(rho_xy_pos['rho_xy'], errors='coerce').notnull()]
    rho_xy_neg = data_T.loc[(data_T['Field'] < 0)].sort_values(by = 'Field', ascending = False)
    rho_xy_neg = rho_xy_neg[pd.to_numeric(rho_xy_neg['rho_xy'], errors='coerce').notnull()]
    
    # interpolate data between corresponding positive and negative fields
    pos_interp = PchipInterpolator(*resolve_monotone(np.concatenate([[H_min], rho_xx_pos['Field']]), np.concatenate([[rho_xx_0], rho_xx_pos['rho_xx']])), extrapolate = False)
    rho_xx_pos_new = pos_interp(Bs)
    neg_interp = PchipInterpolator(*resolve_monotone(np.concatenate([[H_min], np.abs(rho_xx_neg['Field'])]), np.concatenate([[rho_xx_0], rho_xx_neg['rho_xx']])), extrapolate = False)
    rho_xx_neg_new = neg_interp(Bs)

    pos_interp = PchipInterpolator(*resolve_monotone(np.concatenate([[H_min], rho_xy_pos['Field']]), np.concatenate([[rho_xy_0], rho_xy_pos['rho_xy']])), extrapolate = False)
    rho_xy_pos_new = pos_interp(Bs)
    neg_interp = PchipInterpolator(*resolve_monotone(np.concatenate([[H_min], np.abs(rho_xy_neg['Field'])]), np.concatenate([[rho_xy_0], rho_xy_neg['rho_xy']])), extrapolate = False)
    rho_xy_neg_new = neg_interp(Bs)
    
    # get unbiased resistivity
    rho_xx = (rho_xx_pos_new + rho_xx_neg_new)/2
    rho_xxs.append(rho_xx)

    rho_xy = (rho_xy_pos_new - rho_xy_neg_new)/2
    rho_xys.append(rho_xy)

Ts = np.array(Ts)
rho_xxs = np.stack(rho_xxs)
rho_xys = np.stack(rho_xys)

base = 1000
colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)
S_scale = 1E6
q = 1.602E-19

f, ax = plt.subplots(1, 1, figsize=(8,7))
for T, rhoxx in zip(np.flip(Ts, axis = 0), np.flip(rho_xxs, axis = 0)):
    ax.plot(Bs, rhoxx / 1E-2, '-', color = colors[int(T)-2], linewidth = 3)

ax.set_xlabel(r'$B$ [T]', fontsize = 30)
ax.set_ylabel(r'$\rho_{xx}$ [$\Omega$ cm]', fontsize = 30)
ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
ax.set_xlim((np.min(Bs), np.max(Bs)))
ax.yaxis.get_offset_text().set_size(26)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)

cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)))
cb.ax.set_title(r'$T$ [K]', fontsize = 30)
cb.ax.tick_params(length = 5, width = 1.5, labelsize = 26, which = 'both', direction = 'out')
cb.ax.set_yscale('log')
cb.ax.set_yticks([10, 100])
cb.ax.set_yticklabels([10, 100])

f.tight_layout()

f.savefig('rhoxx_vs_B_S3.png')
plt.close()

f, ax = plt.subplots(1, 1, figsize=(8,7))
for T, rhoxy in zip(np.flip(Ts, axis = 0), np.flip(rho_xys, axis = 0)):
    ax.plot(Bs, rhoxy / 1E-2, '-', color = colors[int(T)-2], linewidth = 3)

ax.set_xlabel(r'$B$ [T]', fontsize = 30)
ax.set_ylabel(r'$\rho_{xy}$ [$\Omega$ cm]', fontsize = 30)
ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 26)
ax.set_xlim((np.min(Bs), np.max(Bs)))
ax.yaxis.get_offset_text().set_size(26)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)

cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)))
cb.ax.set_title(r'$T$ [K]', fontsize = 30)
cb.ax.tick_params(length = 5, width = 1.5, labelsize = 26, which = 'both', direction = 'out')
cb.ax.set_yscale('log')
cb.ax.set_yticks([10, 100])
cb.ax.set_yticklabels([10, 100])

f.tight_layout()

f.savefig('rhoxy_vs_B_S3.png')
plt.close()