import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

pris_file = 'D:/python_project/H-TaP/02_Computational/SuperCell/pristine.csv'
dope_file = 'D:/python_project/H-TaP/02_Computational/SuperCell/withH.csv'

pris_data = np.genfromtxt(pris_file, delimiter = ',', skip_header = 1)[:, -1]
dope_data = np.genfromtxt(dope_file, delimiter = ',', skip_header = 1)[:, -1]

fig, ax = plt.subplots(1, 1, figsize = (8, 7))
ax.plot(dope_data, 'o', color = '#2F349A', markerfacecolor = 'w', markersize = 14, markeredgewidth = 3)
ax.axhline(y = 0, linewidth = 2, color = 'k', linestyle = (0, (8, 10)))

ax.set_ylabel(r'$E-E_f$ [eV]', fontsize = 20)
ax.set_xticks([])
ax.set_ylim((-0.3, 0.6))
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 20)

fig.tight_layout()

axin = inset_axes(ax, width = '100%', height = '100%', bbox_to_anchor=(0.05, .6, .45, .35), bbox_transform = ax.transAxes, loc = 3)
axin.plot(dope_data*1000, 'o', color = '#2F349A', markerfacecolor = 'w', markersize = 14, markeredgewidth = 3)
axin.axhline(y = 0, linewidth = 2, color = 'k', linestyle = (0, (8, 10)))

axin.set_ylabel(r'$E-E_f$ [meV]', fontsize = 20)
axin.set_xticks([])
axin.set_ylim((-50, 20))
axin.set_xlim((80, 100))
axin.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 20)
axin.yaxis.set_label_position('right')
axin.yaxis.tick_right()

ax.add_patch(Rectangle((80, -0.05), 20, 0.07, edgecolor = 'k', fill = False, lw = 2))
ax.add_patch(Rectangle((2.8, 0.248), 92.1, 0.318, edgecolor = 'k', fill = False, lw = 2))

l1 = [(80, -0.05), (2.8, 0.248)]
l2 = [(80, 0.02), (2.8, 0.566)]
l3 = [(100, -0.05), (94.9, 0.25)]
l4 = [(100, 0.02), (94.9, 0.566)]
lc = LineCollection([l1, l2, l3, l4], color = 'k', linewidth = 2)
ax.add_collection(lc)

ax.annotate(r'One H in Ta$_{16}$P$_{16}$', (80, -0.25), fontsize = 20)

plt.savefig('dope_weyl.png', dpi = 300)

fig, ax = plt.subplots(1, 1, figsize = (8, 7))
ax.plot(pris_data*1000, 'o', color = '#E33119', markerfacecolor = 'w', markersize = 14, markeredgewidth = 3)
ax.axhline(y = 0, linewidth = 2, color = 'k', linestyle = (0, (8, 10)))

ax.set_ylabel(r'$E-E_f$ [meV]', fontsize = 20)
ax.set_ylim((-50, 20))
ax.set_xticks([])
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 20)

fig.tight_layout()

ax.annotate(r'Ta$_{2}$P$_{2}$', (15, -46), fontsize = 20)

plt.savefig('pris_weyl.png', dpi = 300)