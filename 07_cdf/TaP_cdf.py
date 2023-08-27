import numpy as np
import matplotlib.pyplot as plt
import math

# # with out SOC
# suffix = '_w_wo_s_u_soc'

# with SOC
suffix = '_w_s_soc_wo_u'

scf_file = f'./TaP_scf{suffix}.out'
dos_file = f'./TaP_dos{suffix}.dat'
band_file= f'./TaP_band{suffix}.dat'

zoom_range = 0.1

Ef = None
V = None
with open(scf_file, 'r') as f:
    for line in f.readlines():
        if 'the Fermi energy is' in line:
            Ef = float(line.split()[-2])
        if 'unit-cell volume' in line:
            V = float(line.split()[-2])*0.529177249**3

print(f'Name: TaP{suffix}')
print(f'Fermi\'s energy = {Ef} eV')
print(f'Unit cell volume = {round(V, 3)} cubic angstrom')

dos_data = np.loadtxt(dos_file, unpack = True)
E_dos = dos_data[0] - Ef
DoS = dos_data[1]
CDF = np.cumsum(DoS)*(E_dos[1]-E_dos[0])

def plot_cdf(E_dos, DoS, CDF, V, suffix, zoom_range = None):
    CDF_f = CDF[np.argmin(abs(E_dos))]
    scale = 1
    Eunit = 'eV'
    if zoom_range != None:
        n_min = np.argmin(abs(E_dos + zoom_range/2))
        n_max = np.argmin(abs(E_dos - zoom_range/2))
        E_dos = E_dos[n_min:n_max+1]
        DoS = DoS[n_min:n_max+1]
        CDF = CDF[n_min:n_max+1]
        scale = 1000
        Eunit = 'meV'

    fig, (ax0, ax1) = plt.subplots(2, 1, sharex = True, figsize = (14, 14), dpi = 300)
    fig.subplots_adjust(hspace=0)
    ax2 = ax1.twinx()
    def convert_ax2(ax1):
        y1, y2 = ax1.get_ylim()
        ax2.set_ylim(y1/V*1E24, y2/V*1E24)
        ax2.figure.canvas.draw()    

    ax1.callbacks.connect("ylim_changed", convert_ax2)

    ax0.plot(E_dos*scale, DoS, 'r-', linewidth = 2)
    ax0.axvline(x = 0, linewidth = 1, color = 'k', linestyle = (0, (8, 10)))
    ax0.fill_between(E_dos*scale, 0, DoS, where = (E_dos <= 0), facecolor = 'red', alpha = 0.25)

    ax1.plot(E_dos*scale, CDF - CDF_f, 'r-', linewidth = 2)
    ax1.axvline(x = 0, linewidth = 1, color = 'k', linestyle = (0, (8, 10)))
    ax1.axhline(y = 0, linewidth = 1, color = 'k', linestyle = (0, (8, 10)))

    ax0.set_xlim(min(E_dos)*scale, max(E_dos)*scale)
    ax0.set_ylabel('DOS []', fontsize = 24)
    ax0.tick_params(axis = 'both', which = 'both', width = 2, length = 10, direction = 'in', labelsize = 24)

    ax1.set_xlabel(r'$E-E_f$ ' + f'[{Eunit}]', fontsize = 24)
    ax1.set_ylabel(r'CDF - CDF$_f$ []', fontsize = 24)
    ax1.tick_params(axis = 'both', which = 'both', width = 2, length = 10, direction = 'in', labelsize = 24)

    ax2.set_ylabel(r'Doped Electron [cm$^{-3}$]', fontsize = 24)
    ax2.tick_params(axis = 'both', which = 'both', width = 2, length = 10, direction = 'in', labelsize = 24)
    ax2.yaxis.get_offset_text().set_size(24)
    plt.savefig(f'TaP_dos{suffix}.png')

plot_cdf(E_dos, DoS, CDF, V, suffix)
plot_cdf(E_dos, DoS, CDF, V, suffix+'_zoom', zoom_range = zoom_range)

labels=['$\Gamma$', 'X', 'P', 'N', '$\Gamma$', 'M', 'S', 'S$_0$', '$\Gamma$', 'X', 'R', 'G', 'M']

ks = []
bs = []
with open(band_file, 'r') as f:
    k = None
    b = []
    for i, line in enumerate(f.readlines()):
        if i == 0:
            contents = line.split()
            nb = int(contents[-4][:-1])
        else:
            contents = [float(x) for x in line.split()]
            if len(contents) == 0:
                break
            if k == None:
                k = contents.copy()
            elif len(b) < nb:
                b += contents.copy()
            else:
                ks.append(k)
                bs.append(b)
                k = contents.copy()
                b = []
ks = np.array(ks)
bs = np.array(bs).T - Ef
k_point_diff = np.array([0]+[np.linalg.norm(dk) for dk in ks[1:] - ks[:-1]])
k = np.cumsum(k_point_diff)

N = math.floor((len(ks)-1)/(len(labels)-1))
tick_pos = [k[idx] for idx in range(0, len(k), N)]

plt.figure(dpi = 300)
for b in bs:
    plt.plot(k, b, linewidth = 1, alpha = 1, color = 'k')
plt.xlim((np.min(k), np.max(k)))
for pos in tick_pos:
    plt.axvline(pos, linewidth = 1, color='k', alpha=0.5)
plt.axhline(0, linewidth = 1, color = 'k', alpha=1, linestyle = (0, (8, 10)))
plt.xticks(ticks = tick_pos, labels = labels)
plt.ylabel(r'$E-E_f$ [eV]')
plt.ylim((min(E_dos), max(E_dos)))
plt.savefig(f'TaP_band{suffix}.png')