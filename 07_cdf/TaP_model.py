import numpy as np
import matplotlib.pyplot as plt
import math

dE = np.linspace(-50, 50, 1001)
pi = math.pi

ec = 8E15/(3*pi**2)*(2*((dE-24)**3+24**3)/1.9+((dE+40)**3-40**3)/1.1)
print(8E15/(3*pi**2)*(2*((24-24)**3+24**3)/1.9+((24+40)**3-40**3)/1.1))
print(8E15/(3*pi**2)*(2*((-40-24)**3+24**3)/1.9+((-40+40)**3-40**3)/1.1))

fig, ax = plt.subplots(1, 1, figsize = (14, 7), dpi=300)
ax.plot(dE, ec, 'r-')
ax.axvline(x = 0, linewidth = 1, color = 'k', linestyle = (0, (8, 10)))
ax.axhline(y = 0, linewidth = 1, color = 'k', linestyle = (0, (8, 10)))
ax.set_xlabel(r'$E-E_f$ [meV]', fontsize = 24)
ax.set_ylabel(r'Doped Electron [cm$^{-3}$]', fontsize = 24)
ax.tick_params(axis = 'both', which = 'both', width = 2, length = 10, direction = 'in', labelsize = 24)
ax.yaxis.get_offset_text().set_size(24)
# plt.show()
plt.savefig('TaP_model.png')