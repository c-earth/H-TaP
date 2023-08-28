import matplotlib.pyplot as plt
import numpy as np

kpoints, etot = np.loadtxt('TaP_conv_05_etot_vs_kpoints.dat', delimiter = ' ', unpack = True)

fig, ax = plt.subplots(1, 1, figsize = (8, 7))
ax.plot(kpoints, etot, 'o-', color = '#2F349A', linewidth = 3, markersize = 10)

ax.set_xlabel('kpoints []', fontsize = 30)
ax.set_ylabel(r'$E_{total}$ [Ry]', fontsize = 30)
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax.yaxis.get_offset_text().set_size(26)

fig.tight_layout()
ax.annotate('(d)', xy=(.025, .975), xycoords = 'figure fraction', horizontalalignment = 'left', verticalalignment = 'top', fontsize = 30)

plt.savefig('TaP_conv_05_etot_vs_kpoints.png')