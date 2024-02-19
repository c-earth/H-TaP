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
ax.plot(dope_data*1000, 'o', color = '#2F349A', markerfacecolor = 'w', markersize = 10, markeredgewidth = 3)
ax.axhline(y = 0, linewidth = 2, color = 'k', linestyle = (0, (8, 10)))

ax.set_ylabel(r'$E-E_f$ [meV]', fontsize = 30)
ax.set_xlabel('Index', fontsize = 30)
# ax.set_xticks([])
ax.set_ylim((-300, 600))
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)

axin = inset_axes(ax, width = '100%', height = '100%', bbox_to_anchor=(0.049, .6, .45, .35), bbox_transform = ax.transAxes, loc = 3)
axin.plot(dope_data*1000, 'o', color = '#2F349A', markerfacecolor = 'w', markersize = 10, markeredgewidth = 3)
axin.axhline(y = 0, linewidth = 2, color = 'k', linestyle = (0, (8, 10)))

axin.yaxis.set_label_position('right')
axin.set_ylabel(r'$E-E_f$ [meV]', fontsize = 30, rotation=-90)
axin.set_ylim((-50, 20))
axin.set_xlim((80, 100))
axin.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)
axin.yaxis.tick_right()
axin.yaxis.labelpad = 30

ax.add_patch(Rectangle((80, -50), 20, 70, edgecolor = 'k', fill = False, lw = 2))
ax.add_patch(Rectangle((2.8, 248), 92.1, 318, edgecolor = 'k', fill = False, lw = 2))

l1 = [(80, -50), (2.8, 248)]
l2 = [(80, 20), (2.8, 566)]
l3 = [(100, -50), (94.9, 248)]
l4 = [(100, 20), (94.9, 566)]
lc = LineCollection([l1, l2, l3, l4], color = 'k', linewidth = 2)
ax.add_collection(lc)

fig.subplots_adjust(left = 0.20)
fig.subplots_adjust(right = 0.95)
fig.subplots_adjust(top = 0.95)
fig.subplots_adjust(bottom = 0.12)

plt.savefig('dope_weyl.png', dpi = 260)

fig, ax = plt.subplots(1, 1, figsize = (8, 7))
ax.plot(pris_data*1000, 'o', color = '#E33119', markerfacecolor = 'w', markersize = 10, markeredgewidth = 3)
ax.axhline(y = 0, linewidth = 2, color = 'k', linestyle = (0, (8, 10)))

ax.set_ylabel(r'$E-E_f$ [meV]', fontsize = 30)
ax.set_ylim((-50, 20))
ax.set_xlabel('Index', fontsize = 30)
# ax.set_xticks([])
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)

fig.subplots_adjust(left = 0.20)
fig.subplots_adjust(right = 0.95)
fig.subplots_adjust(top = 0.95)
fig.subplots_adjust(bottom = 0.12)

plt.savefig('pris_weyl.png', dpi = 260)