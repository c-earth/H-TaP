#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl

# from brokenaxes import brokenaxes

#%%
# read files
homedir = '/home/rokabe/data1/manasi/two_band/TwoBand'
s2_file = f"{homedir}/S2_two_band_fitting/model4_params.txt"
s4_file =  f"{homedir}/S4_two_band_fitting/model4_params.txt"
ref_file = f'{homedir}/TaP_data.csv'

ref_data = pd.read_csv(ref_file, sep=",")
s2_data = pd.read_csv(s2_file, sep="\t")
s4_data = pd.read_csv(s4_file, sep="\t")

#%%

class Line:
  def __init__(self,Xs,Ys):
    self.Xs = Xs
    self.Ys = Ys
    self.m= (self.Ys[1] - self.Ys[0])/(self.Xs[1] - self.Xs[0])
    self.c= self.Ys[0] - self.m * self.Xs[0]

def findSolution(L1,L2, figplot=False):
    x=(L1.c-L2.c)/(L2.m-L1.m)
    y=L1.m*x+L1.c
    if figplot:
        X=[x for x in range(-10,11)]
        Y1=[(L1.m*x)+L1.c for x in X]
        Y2=[(L2.m*x)+L2.c for x in X]
        plt.plot(X,Y1,'-r',label=f'y={L1.m}x+{L1.c}')
        plt.plot(X,Y2,'-b',label=f'y={L2.m}x+{L2.c}')
        plt.legend(loc='upper right')
        plt.show()
    return (x,y)

def findintersection2(XX, YYe, YYh):
    pos = np.where(YYe-YYh <= 0)[-1]
    print(pos)
    i0 = pos[-1]
    i1 = i0 + 1
    A, B= YYe[i0], YYe[i1]
    C, D= YYh[i0], YYh[i1]
    X0 = XX[i0]
    X1 = XX[i1]
    y1 = np.array([A, B])
    y2 = np.array([C, D])
    x = np.array([X0, X1])
    L1 = Line(x, y1)
    L2 = Line(x, y2)
    return findSolution(L1,L2)

#%%
fig_name = 'fig4'
len_unit = "cm"
per_volume_convert = {'m': 1, "cm": 1e-06}    # 1m^(-3) = 1e-06 cm^(-3)
len_sq_convert = {'m': 1, "cm": 1e+04}    # 1m^(2) = 1e+04 cm^(2)
cnst1 = per_volume_convert[len_unit]
cnst2 = len_sq_convert[len_unit]
fig = plt.figure(figsize=(25,30))
gs = fig.add_gridspec(3,2, hspace=0)
axs= gs.subplots(sharex=True)
cmap = plt.cm.get_cmap('gnuplot')   # same color map as the other figures
color_indices = [90, 220]   # adjust to 
colors = [cmap(idx) for idx in color_indices]

axA = axs[0,0]
axA.plot(ref_data['T_nh'], ref_data['n_h'],linestyle='-',marker='o',markersize=14,linewidth=3,label='$n_h$', color=colors[0])
axA.plot(ref_data['T_ne'], ref_data['n_e'],linestyle='-',marker='o',markersize=14,linewidth=3,label='$n_e$', color=colors[1])
axA.set_ylabel(f'Career density $({len_unit}^{{-3}})$', fontsize=24)
axA.set_yscale('log')
legA = axA.legend(fontsize=24, loc='lower right')
legA.get_frame().set_linewidth(0.0)
axA.tick_params(axis='both', which='both', direction='in', labelsize=24, width=3, length=10)
axA.yaxis.get_offset_text().set_size(24)
TsA, YeA, YhA = ref_data['T_ne'], ref_data['n_e'], ref_data['n_h'][::-1].reset_index(drop=True)
posA = findintersection2(TsA, YeA, YhA)
print('posA: ', posA)
axA.vlines(x=[posA[0]], ymin=[0], ymax=[posA[1]], colors='gray', ls='--', lw=3)

axB = axs[0,1] 
axB.plot(ref_data['T_uh'], ref_data['u_h'],linestyle='-',marker='o',markersize=14,linewidth=3,label='$\mu_h$', color=colors[0])
axB.plot(ref_data['T_ue'], ref_data['u_e'],linestyle='-',marker='o',markersize=14,linewidth=3,label='$\mu_e$', color=colors[1])
axB.set_ylabel(f'Mobility  $({len_unit}^2V^{{-1}}s^{{-1}})$', fontsize=24)
legB = axB.legend(fontsize=24, loc='upper right')
legB.get_frame().set_linewidth(0.0)
axB.tick_params(axis='both', which='both', direction='in', labelsize=24, width=3, length=10)
axB.yaxis.get_offset_text().set_size(24)
fig.patch.set_facecolor('white')


ax1 = axs[2,0] 
ax1.plot(s2_data['T(K)'], s2_data['nh(m^-3)']*cnst1,linestyle='-',marker='o',markersize=14,linewidth=3,label='$n_h$', color=colors[0])
ax1.plot(s2_data['T(K)'], s2_data['ne(m^-3)']*cnst1,linestyle='-',marker='o',markersize=14,linewidth=3,label='$n_e$', color=colors[1])
ax1.set_ylabel(f'Career density $({len_unit}^{{-3}})$', fontsize=24)
ax1.set_yscale('log')
leg1 = ax1.legend(fontsize=24, loc='lower right')
leg1.get_frame().set_linewidth(0.0)
ax1.tick_params(axis='both', which='both', direction='in', labelsize=24, width=3, length=10)
ax1.yaxis.get_offset_text().set_size(24)
Ts2, Ye2, Yh2 = s2_data['T(K)'], s2_data['ne(m^-3)']*cnst1, s2_data['nh(m^-3)']*cnst1
pos1 = findintersection2(Ts2, Ye2, Yh2)
print('pos1: ', pos1)
ax1.vlines(x=[pos1[0]], ymin=[0], ymax=[pos1[1]], colors='gray', ls='--', lw=3)

ax2 = axs[2,1] 
ax2.plot(s2_data['T(K)'], s2_data['uh(m^2/Vs)']*cnst2,linestyle='-',marker='o',markersize=14,linewidth=3,label='$\mu_h$', color=colors[0])
ax2.plot(s2_data['T(K)'], s2_data['ue(m^2/Vs)']*cnst2,linestyle='-',marker='o',markersize=14,linewidth=3,label='$\mu_e$', color=colors[1])
ax2.set_ylabel(f'Mobility  $({len_unit}^2V^{{-1}}s^{{-1}})$', fontsize=24)
leg2 = ax2.legend(fontsize=24, loc='upper right')
leg2.get_frame().set_linewidth(0.0)
ax2.tick_params(axis='both', which='both', direction='in', labelsize=24, width=3, length=10)
ax2.yaxis.get_offset_text().set_size(24)
fig.patch.set_facecolor('white')

ax3 = axs[1,0] 
ax3.plot(s4_data['T(K)'], s4_data['nh(m^-3)']*cnst1,linestyle='-',marker='o',markersize=14,linewidth=3,label='$n_h$', color=colors[0])
ax3.plot(s4_data['T(K)'], s4_data['ne(m^-3)']*cnst1,linestyle='-',marker='o',markersize=14,linewidth=3,label='$n_e$', color=colors[1])
ax3.set_xlabel(r'$T$ $(K)$', fontsize=24)
ax3.set_ylabel(f'Career density $({len_unit}^{{-3}})$', fontsize=24)
ax3.set_yscale('log')
leg3 = ax3.legend(fontsize=24, loc='lower right')
leg3.get_frame().set_linewidth(0.0)
ax3.tick_params(axis='both', which='both', direction='in', labelsize=24, width=3, length=10)
ax3.set_ylim([0.95*10**(19), 3*10**(19.1)])
ax3.xaxis.get_offset_text().set_size(24)
ax3.yaxis.get_offset_text().set_size(24)
Ts4, Ye4, Yh4 = s4_data['T(K)'], s4_data['ne(m^-3)']*cnst1, s4_data['nh(m^-3)']*cnst1
pos3 = findintersection2(Ts4, Ye4, Yh4)
print('pos3', pos3)
ax3.vlines(x=[pos3[0]], ymin=[0], ymax=[pos3[1]], colors='gray', ls='--', lw=2.6)

ax4 = axs[1,1]
ax4.plot(s4_data['T(K)'], s4_data['uh(m^2/Vs)']*cnst2,linestyle='-',marker='o',markersize=14,linewidth=3,label='$\mu_h$', color=colors[0])
ax4.plot(s4_data['T(K)'], s4_data['ue(m^2/Vs)']*cnst2,linestyle='-',marker='o',markersize=14,linewidth=3,label='$\mu_e$', color=colors[1])
ax4.set_xlabel(r'$T$ $(K)$', fontsize=24)
ax4.set_ylabel(f'Mobility $({len_unit}^2V^{{-1}}s^{{-1}})$', fontsize=24)
leg4 = ax4.legend(fontsize=24, loc='upper right')
leg4.get_frame().set_linewidth(0.0)
ax4.tick_params(axis='both', which='both', direction='in', labelsize=24, width=3, length=10)
ax4.xaxis.get_offset_text().set_size(24)
ax4.yaxis.get_offset_text().set_size(24)
fig.patch.set_facecolor('white')
fig.savefig(fig_name + '.pdf')
fig.savefig(fig_name + '.png')

#%%
