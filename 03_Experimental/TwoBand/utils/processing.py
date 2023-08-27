import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

q = 1.602E-19

def read_file(filename, Hmin = 0, Hmax = -1):
    B = []
    S = []
    with open(filename, 'r') as f:
        for line in f.readlines()[1:]:
            line_data =  line.split(',') 
            B.append(float(line_data[1]))
            S.append(float(line_data[2]))

    B = np.array(B)
    S = np.array(S)

    idxs = np.argsort(B)
    B = B[idxs]
    S = S[idxs]
    Lmin = int(Hmin/max(B) * len(B))
    if Hmax > Hmin:
        
        Lmax = int(Hmax/max(B) * len(B))

    else:
        Lmax = len(B)

    B = B[Lmin: Lmax]
    S = S[Lmin: Lmax]
    return B, S

def read_params(filename):
    with open(filename, 'r') as f:
        params = []
        for line in f.readlines()[1:]:
            params.append([float(x) for x in line.split('\t')])
    return np.array(params)

def plot_prediction(savename, T, model, BsSs, p, S_scale):
    Ss_pred = model(BsSs, *p)
    l = int(len(BsSs)/4)
    Bxx = BsSs[0*l:1*l]
    Bxy = BsSs[1*l:2*l]
    Sxx_true = BsSs[2*l:3*l] * S_scale
    Sxy_true = BsSs[3*l:4*l] * S_scale
    Sxx_pred = Ss_pred[0*l:1*l] * S_scale
    Sxy_pred = Ss_pred[1*l:2*l] * S_scale
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
    
    ax1.plot(Bxx, Sxx_true*1E-2, linestyle='', marker='.', markersize=14, linewidth=3, color='#2F349A', label=f'expected $Sxx$, {T}K')
    ax1.plot(Bxx, Sxx_pred*1E-2, linestyle='--', marker='', markersize=14, linewidth=3, color='#E33119', label=f'predicted $Sxx$, {T}K')
    
    ax2.plot(Bxy, Sxy_true*1E-2, linestyle='', marker='.', markersize=14, linewidth=3, color='#2F349A', label=f'expected $Sxy$, {T}K')
    ax2.plot(Bxy, Sxy_pred*1E-2, linestyle='--', marker='', markersize=14, linewidth=3, color='#E33119', label=f'predicted $Sxy$, {T}K')
    
    ax1.set_xlabel(r'$B$ [T]', fontsize=20)
    ax1.set_ylabel(r'$\sigma_{xx}$ [$\Omega^{-1}$cm$^{-1}$]', fontsize=20)
    ax1.legend(fontsize=20,loc='best')
    ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
    ax1.xaxis.get_offset_text().set_size(20)
    ax1.yaxis.get_offset_text().set_size(20)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText = True)

    ax2.set_xlabel(r'$B$ [T]', fontsize=20)
    ax2.set_ylabel(r'$\sigma_{xy}$ [$\Omega^{-1}$cm$^{-1}$]', fontsize=20)
    ax2.legend(fontsize=20,loc='best')
    ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
    ax2.xaxis.get_offset_text().set_size(20)
    ax2.yaxis.get_offset_text().set_size(20)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText = True)

    f.tight_layout()
    
    f.savefig(savename, dpi=300)
    plt.close()

def plot_residual(savename, params_files):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
    for params_file in params_files:
        params = read_params(params_file)
        name = params_file.split('/')[-1][:-11]
        ax1.plot(params[:, 0], params[:, -4]*1E-2, linestyle='-', marker='.', markersize=14, linewidth=3, label=name, color = '#2F349A')
        ax2.plot(params[:, 0], params[:, -3]*1E-2, linestyle='-', marker='.', markersize=14, linewidth=3, label=name, color = '#2F349A')

    ax1.set_xlabel(r'$T$ [K]', fontsize=20)
    ax1.set_ylabel(r'RMSE of Fitted $\sigma_{xx}$ [$\Omega^{-1}$cm$^{-1}$]', fontsize=20)
    ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
    ax1.xaxis.get_offset_text().set_size(20)
    ax1.yaxis.get_offset_text().set_size(20)
    ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)

    ax2.set_xlabel(r'$T$ $[K]$', fontsize=20)
    ax2.set_ylabel(r'RMSE of Fitted $\sigma_{xy}$ [$\Omega^{-1}$cm$^{-1}$]', fontsize=20)
    ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
    ax2.xaxis.get_offset_text().set_size(20)
    ax2.yaxis.get_offset_text().set_size(20)
    ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)

    f.tight_layout()
    f.savefig(savename, dpi=300)
    plt.close()

def plot_fitting_params(savename, params_files):
    for params_file in params_files:
        params = read_params(params_file)
        name = params_file.split('/')[-1][:-11]

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
            
        ax1.plot(params[:, 0], params[:, 7]*1E-6,linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_e$', color = '#E33119')
        ax1.plot(params[:, 0], params[:, 3]*1E-6,linestyle='--',marker='o',markersize=14,linewidth=3,label='$n_h$', color = '#2F349A')
        ax2.plot(params[:, 0], params[:, 5]*1E4,linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_e$', color = '#E33119')
        ax2.plot(params[:, 0], params[:, 1]*1E4,linestyle='--',marker='s',markersize=14,linewidth=3,label='$\mu_h$', color = '#2F349A')

        ax1.set_xlabel(r'$T$ [K]', fontsize=20)
        ax1.set_ylabel(r'$n$ [cm$^{-3}$]', fontsize=20)
        ax1.legend(fontsize=20, loc='center left')
        ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
        ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
        ax1.xaxis.get_offset_text().set_size(20)
        ax1.yaxis.get_offset_text().set_size(20)

        ax2.set_xlabel(r'$T$ [K]', fontsize=20)
        ax2.set_ylabel(r'$\mu$ (cm$^2$V$^{-1}$s$^{-1}$)', fontsize=20)
        ax2.legend(fontsize=20, loc='center right')
        ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
        ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
        ax2.xaxis.get_offset_text().set_size(20)
        ax2.yaxis.get_offset_text().set_size(20)
        
        f.tight_layout()
        
        f.savefig(savename[:-11] + name + savename[-11:], dpi=300)
        plt.close()

def plot_sigma(Ts, Bxxs, Sxxs, Bxys, Sxys, resu_dir, S_scale):
    base = 1000
    colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
    for T, Bxx, Sxx, Bxy, Sxy in zip(np.flip(Ts, axis = 0), np.flip(Bxxs, axis = 0), np.flip(Sxxs, axis = 0), np.flip(Bxys, axis = 0), np.flip(Sxys, axis = 0)):
        ax1.plot(Bxx, Sxx * S_scale * 1E-2, '-', color = colors[int(T)-2], linewidth = 3)
        ax2.plot(Bxy, Sxy * S_scale * 1E-2, '-', color = colors[int(T)-2], linewidth = 3)

    ax1.set_xlabel(r'$B$ [T]', fontsize = 20)
    ax1.set_ylabel(r'$\sigma_{xx}$ [$\Omega^{-1}$cm$^{-1}$]', fontsize = 20)
    ax1.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    ax1.set_xlim((np.min(Bxx), np.max(Bxx)))
    ax1.yaxis.get_offset_text().set_size(20)
    ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)

    ax2.set_xlabel(r'$B$ [T]', fontsize = 20)
    ax2.set_ylabel(r'$\sigma_{xy}$ [$\Omega^{-1}$cm$^{-1}$]', fontsize = 20)
    ax2.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    ax2.set_xlim((np.min(Bxy), np.max(Bxy)))
    ax2.yaxis.get_offset_text().set_size(20)
    ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)))
    cb.ax.set_title(r'$T$ [K]', fontsize = 20)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 20, which = 'both', direction = 'out')
    cb.ax.set_yscale('log')
    cb.ax.set_yticks([10, 100])
    cb.ax.set_yticklabels([10, 100])

    f.tight_layout()

    f.savefig(os.path.join(resu_dir, 'sigma_vs_B.png'))
    plt.close()

def plot_relative_residual(savename, params_files):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,7))
    for params_file in params_files:
        params = read_params(params_file)
        name = params_file.split('/')[-1][:-11]
        ax1.plot(params[:, 0], params[:, -2], linestyle='-', marker='.', markersize=14, linewidth=3, label=name, color = '#2F349A')
        ax2.plot(params[:, 0], params[:, -1], linestyle='-', marker='.', markersize=14, linewidth=3, label=name, color = '#2F349A')

    ax1.set_xlabel(r'$T$ [K]', fontsize=20)
    ax1.set_ylabel(r'RMSRE of Fitted $\sigma_{xx}$ []', fontsize=20)
    ax1.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
    ax1.xaxis.get_offset_text().set_size(20)
    ax1.yaxis.get_offset_text().set_size(20)
    ax1.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)

    ax2.set_xlabel(r'$T$ $(K)$', fontsize=20)
    ax2.set_ylabel(r'RMSRE of Fitted $\sigma_{xy}$ []', fontsize=20)
    ax2.tick_params(axis='both', which='both', direction='in', labelsize=20, width=1.5, length=5)
    ax2.xaxis.get_offset_text().set_size(20)
    ax2.yaxis.get_offset_text().set_size(20)
    ax2.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)

    f.tight_layout()

    f.savefig(savename, dpi=300)
    plt.close()