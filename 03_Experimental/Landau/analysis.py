import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from util.Kohler import plot_rho, plot_MR, plot_MRK, plot_MREK
from util.back_ground import subbg_po, subbg_pp, subbg_de
from util.Landau import FFT, FFT_peaks, signal_filter

# # MR_S2
# data_file = 'D:/python_project/H-TaP/03_Experimental/Landau/data/MR_S2_extracted.pkl'
# params_file = 'D:/python_project/H-TaP/03_Experimental/Landau/data/MR_S2_params.txt'
# resu_dir = 'D:/python_project/H-TaP/03_Experimental/Landau/results/MR_S2/'
# po_power = 7
# pieces = 2
# pp_power = 4
# avg_window = 10
# T_max = 12

# MR_S4
data_file = 'D:/python_project/H-TaP/03_Experimental/Landau/data/MR_S4_extracted.pkl'
params_file = 'D:/python_project/H-TaP/03_Experimental/Landau/data/MR_S4_params.txt'
resu_dir = 'D:/python_project/H-TaP/03_Experimental/Landau/results/MR_S4/'
po_power = 5
pieces = 2
pp_power = 3
avg_window = 10
T_max = 22

# select background options
po = True
pp = True
de = True

q_min = 0
q_max = 30
r_min = 1
r_max = 30

# create analysis result folder
if not os.path.exists(resu_dir):
    os.mkdir(resu_dir)

# load extracted data
with open(data_file, 'rb') as f:
    Ts, Bs, rho_xxs = pkl.load(f)

# calculate magnetic resistances from resistivities
MRs = (rho_xxs / rho_xxs[:, :1] - 1) * 100

# Kohler's rule analysis
cutoff = 10
plot_rho(Ts, Bs, rho_xxs, cutoff, resu_dir)
plot_MR(Ts, Bs, MRs, cutoff, resu_dir)
# plot_MRK(Ts, Bs, MRs, rho_xxs[:, :1], cutoff, resu_dir)
# plot_MREK(Ts, Bs, MRs, rho_xxs[:, :1], cutoff, resu_dir, params_file = params_file)

# Bs = Bs[10:]
# MRs = MRs[:, 10:]
# # background subtraction
# QO_data = dict()
# if po:
#     QO_data['po'] = subbg_po(Ts, Bs, MRs, po_power, T_max, resu_dir)
# if pp:
#     QO_data['pp'] = subbg_pp(Ts, Bs, MRs, pp_power, pieces, T_max, resu_dir)
# if de:
#     QO_data['de'] = subbg_de(Ts, Bs, MRs, avg_window, T_max, resu_dir)

# base = 1000
# colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
# f, ax = plt.subplots(1, 1, figsize = (8, 2.5))

# for subbg, (Ts_sub, Hs_sub, MRs_sub) in QO_data.items():
#     iHs, MRs_iH, qs, MRs_iH_fft = FFT(Ts_sub, Hs_sub[Hs_sub > 0], MRs_sub[:, Hs_sub > 0], T_max, q_min, q_max, subbg, resu_dir, Ts)
#     for T, q, MR_iH_fft, MR_iH in zip(Ts_sub, qs, MRs_iH_fft, MRs_iH):
#         qpeaks, y_widthes = FFT_peaks(q, MR_iH_fft, r_min, r_max, T_max, resu_dir, T, subbg, Ts)
#         signal_filter(iHs, q, MR_iH_fft, qpeaks, y_widthes, T_max, resu_dir, T, subbg, Ts)

#         ax.plot(qpeaks, T*np.ones(len(qpeaks)), linestyle = '', marker='o', markersize=10, color=colors[int(T)-2])

# ax.set_ylabel(r'$T$ [K]', fontsize=30)
# ax.set_xlabel(r'$q$ [T]', fontsize = 30)
# ax.set_xlim((q_min, q_max))
# ax.set_ylim((0, T_max))
# ax.tick_params(axis='both', which='both', direction='in', labelsize=26, width=1.5, length=5)

# f.tight_layout()

# f.savefig(os.path.join(resu_dir, f'qpeaks_allK.png'), dpi=300)
# plt.close()