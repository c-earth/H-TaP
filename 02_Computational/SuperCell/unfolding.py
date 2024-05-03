from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

path = 2
primitive_file = f'./Path {path}/primitive.mat'
primitive_data = loadmat(primitive_file)

unfold_file = f'./Path {path}/unfold.mat'
unfold_data = loadmat(unfold_file)

E_f = primitive_data['Efermi']
print(E_f)
eigval = primitive_data['eigval']
bs = (eigval - E_f).T
ks = np.arange(len(eigval))

fig, ax = plt.subplots(1, 1, figsize = (8, 7))
ax.axhline(0, linewidth = 2, color = 'k', alpha = 1, linestyle = (0, (8, 10)))
for b in bs:
    ax.scatter(ks, b, s = 20, marker = 'o', alpha = 1, color = '#E33119')
ax.set_xticks([])
ax.set_yticks([-2, -1, 0, 1, 2])
# ax.set_xticks(ticks = tick_pos, labels = labels, fontsize = 30)
ax.set_ylabel(r'$E-E_f$ [eV]', fontsize = 30)
ax.set_xlabel(r'$k$', fontsize = 30)
ax.set_ylim((-3, 3))
ax.set_xlim((0, 49))
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax.yaxis.get_offset_text().set_size(26)

fig.tight_layout()

plt.savefig(f'TaP_band_primitive_{path}.png', dpi = 300)

E_f_u = unfold_data['eref']
print(E_f_u)
dE_f = E_f_u - E_f
print(dE_f)
eigval = unfold_data['sw'][0, :, :, 0]
bs = (eigval - E_f_u).T
ks = np.arange(len(eigval))/2

weights = unfold_data['sw'][0, :, :, 1].T
weights /= np.max(weights)

fig, ax = plt.subplots(1, 1, figsize = (8, 7))
ax.axhline(0, linewidth = 2, color = 'k', alpha = 1, linestyle = (0, (8, 10)))
for b, weight in zip(bs, weights):
    ax.scatter(ks, b, s = 20*weight, marker = 'o', alpha = 1, color = '#2F349A')
ax.set_xticks([])
ax.set_yticks([-3, -2, -1, 0, 1, 2])
# ax.set_xticks(ticks = tick_pos, labels = labels, fontsize = 30)
ax.set_ylabel(r'$E-E_f$ [eV]', fontsize = 30)
ax.set_xlabel(r'$k$', fontsize = 30)
ax.set_ylim((-3-dE_f, 3-dE_f))
ax.set_xlim((0, 50))
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax.yaxis.get_offset_text().set_size(26)

fig.tight_layout()

plt.savefig(f'TaP_band_unfold_{path}.png', dpi = 300)