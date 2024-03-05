import numpy as np
import matplotlib.pyplot as plt
import math

dE = np.linspace(-60, 10, 1001)
pi = math.pi

ec = 8E15/(3*pi**2)*(2*((dE-5)**3+5**3)/1.9+((dE+55)**3-55**3)/1.1)

fig, ax = plt.subplots(1, 1, figsize = (8, 7))
ax.plot(dE, ec, '-', color = '#E33119', linewidth = 3)
ax.axvline(x = 0, linewidth = 2, color = 'k', linestyle = (0, (8, 10)))
ax.axhline(y = 0, linewidth = 2, color = 'k', linestyle = (0, (8, 10)))

ax.set_xlabel(r'$E-E_f$ [meV]', fontsize = 30)
ax.set_ylabel(r'Doped Electron [cm$^{-3}$]', fontsize = 30)
ax.tick_params(axis = 'both', which = 'both', top = False, right = False, width = 1.5, length = 5, direction = 'in', labelsize = 26)
ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText = True)
ax.yaxis.get_offset_text().set_size(26)

fig.tight_layout()
plt.show()
plt.savefig('TaP_model.png', dpi = 300)