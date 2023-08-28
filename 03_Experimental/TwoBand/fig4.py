import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

s2_file = f'./S2_two_band_fitting/model4_params.txt'
s4_file = f'./S4_two_band_fitting/model4_params.txt'
ref_file = f'TaP_data.csv'

ref_data = pd.read_csv(ref_file, sep = ",")
s2_data = pd.read_csv(s2_file, sep = "\t")
s4_data = pd.read_csv(s4_file, sep = "\t")

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

fig_name = 'fig4'
len_unit = "cm"
per_volume_convert = {'m': 1, "cm": 1e-06}    # 1m^(-3) = 1e-06 cm^(-3)
len_sq_convert = {'m': 1, "cm": 1e+04}    # 1m^(2) = 1e+04 cm^(2)
cnst1 = per_volume_convert[len_unit]
cnst2 = len_sq_convert[len_unit]

fig, axs = plt.subplots(3, 2, sharex = True, figsize = (16, 21))

axA = axs[0,0]
axA.plot(ref_data['T_nh'], ref_data['n_h'],linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_h$', color='#2F349A')
axA.plot(ref_data['T_ne'], ref_data['n_e'],linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_e$', color='#E33119')
axA.set_ylabel(r'$n$ [cm$^{-3}$]', fontsize=20)
legA = axA.legend(fontsize=20, loc='lower right')
legA.get_frame().set_linewidth(0.0)
axA.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
axA.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
axA.yaxis.get_offset_text().set_size(20)
TsA, YeA, YhA = ref_data['T_ne'], ref_data['n_e'], ref_data['n_h'][::-1].reset_index(drop=True)
posA = findintersection2(TsA, YeA, YhA)
axA.axvline(x=posA[0], ymax=posA[1], color='k', ls='--', lw=2)
axA.annotate(r'pristine', (0, 4.75E19), fontsize = 20)

axB = axs[0,1] 
axB.plot(ref_data['T_uh'], ref_data['u_h'],linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_h$', color='#2F349A')
axB.plot(ref_data['T_ue'], ref_data['u_e'],linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_e$', color='#E33119')
axB.set_ylabel(r'$\mu$ [cm$^2$V$^{-1}$s$^{-1}$]', fontsize=20)
legB = axB.legend(fontsize=20, loc='upper right')
legB.get_frame().set_linewidth(0.0)
axB.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
axB.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
axB.yaxis.get_offset_text().set_size(20)
axB.annotate(r'pristine', (100, 1.3E5), fontsize = 20)
fig.patch.set_facecolor('white')


ax1 = axs[2,0] 
ax1.plot(s2_data['T(K)'], s2_data['nh(m^-3)']*cnst1,linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_h$', color='#2F349A')
ax1.plot(s2_data['T(K)'], s2_data['ne(m^-3)']*cnst1,linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_e$', color='#E33119')
ax1.set_ylabel('$n$ [cm$^{-3}$]', fontsize=20)
ax1.set_xlabel(r'$T$ [K]', fontsize=20)
leg1 = ax1.legend(fontsize=20, loc='lower right')
leg1.get_frame().set_linewidth(0.0)
ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax1.yaxis.get_offset_text().set_size(20)
Ts2, Ye2, Yh2 = s2_data['T(K)'], s2_data['ne(m^-3)']*cnst1, s2_data['nh(m^-3)']*cnst1
pos1 = findintersection2(Ts2, Ye2, Yh2)
ax1.axvline(x=pos1[0], ymax=pos1[1], color='k', ls='--', lw=2)
ax1.annotate(r'S$_{20m}$', (0, 5.25E19), fontsize = 20)

ax2 = axs[2,1] 
ax2.plot(s2_data['T(K)'], s2_data['uh(m^2/Vs)']*cnst2,linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_h$', color='#2F349A')
ax2.plot(s2_data['T(K)'], s2_data['ue(m^2/Vs)']*cnst2,linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_e$', color='#E33119')
ax2.set_ylabel(r'$\mu$ [cm$^2$V$^{-1}$s$^{-1}$]', fontsize=20)
ax2.set_xlabel(r'$T$ [K]', fontsize=20)
leg2 = ax2.legend(fontsize=20, loc='upper right')
leg2.get_frame().set_linewidth(0.0)
ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax2.yaxis.get_offset_text().set_size(20)
ax2.annotate(r'S$_{20m}$', (100, 9E4), fontsize = 20)
fig.patch.set_facecolor('white')

ax3 = axs[1,0] 
ax3.plot(s4_data['T(K)'], s4_data['nh(m^-3)']*cnst1,linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_h$', color='#2F349A')
ax3.plot(s4_data['T(K)'], s4_data['ne(m^-3)']*cnst1,linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_e$', color='#E33119')
ax3.set_ylabel(r'$n$ [cm$^{-3}$]', fontsize=20)
leg3 = ax3.legend(fontsize=20, loc='lower right')
leg3.get_frame().set_linewidth(0.0)
ax3.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
ax3.set_ylim([0.95*10**(19), 3*10**(19.1)])
ax3.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax3.xaxis.get_offset_text().set_size(20)
ax3.yaxis.get_offset_text().set_size(20)
Ts4, Ye4, Yh4 = s4_data['T(K)'], s4_data['ne(m^-3)']*cnst1, s4_data['nh(m^-3)']*cnst1
pos3 = findintersection2(Ts4, Ye4, Yh4)
ax3.axvline(x=pos3[0], ymax=pos3[1], color='k', ls='--', lw=2)
ax3.annotate(r'S$_{3m}$', (0, 3.5E19), fontsize = 20)

ax4 = axs[1,1]
ax4.plot(s4_data['T(K)'], s4_data['uh(m^2/Vs)']*cnst2,linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_h$', color='#2F349A')
ax4.plot(s4_data['T(K)'], s4_data['ue(m^2/Vs)']*cnst2,linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_e$', color='#E33119')
ax4.set_xlabel(r'$T$ [K]', fontsize=20)
ax4.set_ylabel(r'$\mu$ [cm$^2$V$^{-1}$s$^{-1}$]', fontsize=20)
leg4 = ax4.legend(fontsize=20, loc='upper right')
leg4.get_frame().set_linewidth(0.0)
ax4.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
ax4.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax4.xaxis.get_offset_text().set_size(20)
ax4.yaxis.get_offset_text().set_size(20)
ax4.annotate(r'S$_{3m}$', (100, 4E4), fontsize = 20)
fig.patch.set_facecolor('white')

plt.setp(ax1.get_yaxis().get_offset_text(), visible=False)
plt.setp(ax2.get_yaxis().get_offset_text(), visible=False)
plt.setp(ax3.get_yaxis().get_offset_text(), visible=False)
plt.setp(ax4.get_yaxis().get_offset_text(), visible=False)

fig.tight_layout()
fig.subplots_adjust(hspace = 0)

plt.savefig(fig_name + '.png', dpi = 300)