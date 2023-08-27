import matplotlib.pyplot as plt
import numpy as np

ecutrho, etot = np.loadtxt('TaP_conv_02_etot_vs_ecutrho.dat', delimiter = ' ', unpack = True)

fig, ax = plt.subplots(1, 1, figsize = (8, 7))
ax.plot(ecutrho, etot, '.-', color = '#2F349A', linewidth = 3, markersize = 14)

ax.set_xlabel('ecutrho [Ry]', fontsize = 20)
ax.set_ylabel(r'$E_{total}$ [Ry]', fontsize = 20)
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 20)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax.yaxis.get_offset_text().set_size(20)

fig.tight_layout()

plt.savefig('TaP_conv_02_etot_vs_ecutrho.png')