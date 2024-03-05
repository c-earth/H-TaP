import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import griddata as gd

nx = 20
ny = 20
nz = 60

T = 0.08


plot_coord = np.mgrid[0:0.5:nx*1j, 0:0.5:ny*1j, 0:1:nz*1j]

P_coord = np.array([[0.00, 0.25, 0.25, 0.00], [0.00, 0.25, 0.00, 0.25], [0.5834442871, 0.0834442061, 0.3334467164, 0.8334465196]]).T
T_coord = np.array([[0.00, 0.25, 0.00, 0.25], [0.00, 0.25, 0.25, 0.00], [0.0003553003, 0.5003554192, 0.2503518911, 0.7503518785]]).T

data_raw = np.loadtxt('./H-TaP_monte_carlo_H.dat')[:, 1:]

data_coord = data_raw[:, :-1]
data_value = data_raw[:, -1]

data_coord = np.concatenate([data_coord, np.mod(np.array([0.25, 0.25, 0.50])+data_coord, np.array([0.5, 0.5, 1.0]))])
data_value = np.concatenate([data_value, data_value])

P_coord = np.concatenate([P_coord, np.array([0.5, 0.0, 0.0])+np.array([-1.0, 1.0, 1.0])*P_coord])
T_coord = np.concatenate([T_coord, np.array([0.5, 0.0, 0.0])+np.array([-1.0, 1.0, 1.0])*T_coord])
data_coord = np.concatenate([data_coord, np.array([0.5, 0.0, 0.0])+np.array([-1.0, 1.0, 1.0])*data_coord])
data_value = np.concatenate([data_value, data_value])

P_coord = np.concatenate([P_coord, np.array([0.0, 0.5, 0.0])+np.array([1.0, -1.0, 1.0])*P_coord])
T_coord = np.concatenate([T_coord, np.array([0.0, 0.5, 0.0])+np.array([1.0, -1.0, 1.0])*T_coord])
data_coord = np.concatenate([data_coord, np.array([0.0, 0.5, 0.0])+np.array([1.0, -1.0, 1.0])*data_coord])
data_value = np.concatenate([data_value, data_value])

data_coord = np.concatenate([data_coord, np.mod(np.array([5.0, 0.25, 0.25])+np.array([-1.0, 1.0, 1.0])*data_coord[:, [1, 0, 2]], np.array([0.5, 0.5, 1.0]))])
data_value = np.concatenate([data_value, data_value])


# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# plot = ax.scatter(*(data_coord.T), s=5, c = np.exp(-(data_value-np.min(data_value))/T), cmap = 'seismic', alpha=0.2)
# ax.scatter(*(P_coord.T), s=200, c='k', alpha =0.9)
# ax.scatter(*(T_coord.T), s=200, c='g', alpha =0.9)
# ax.set_box_aspect([1,1,1.7153*2])
# fig.colorbar(plot, ax = ax, shrink = 0.5, aspect = 10)
# plt.show()

shifts = []
for dx in [-0.5, 0.0, 0.5]:
    for dy in [-0.5, 0.0, 0.5]:
        for dz in [-1.0, 0.0, 1.0]:
            shifts.append(np.array([dx, dy, dz]))

V = gd(np.concatenate([data_coord + shift for shift in shifts]), np.concatenate([data_value for _ in shifts]), np.transpose(plot_coord, axes = [1, 2, 3, 0]), method = 'nearest')
V = np.exp(-(V-np.min(V))/T)

xs, ys, zs = plot_coord
vs = V

fig = go.Figure(data=go.Volume(
    x=xs.flatten(),
    y=ys.flatten(),
    z=zs.flatten(),
    value=vs.flatten(),
    isomin=0,
    isomax=0.3,
    opacity=0.2,
    surface_count=1,
    ))
fig.update_scenes(aspectratio=dict(x = 1, y = 1, z = 1.7153*2))
fig.show()

# for index in np.argsort(V.flatten())[-10:]:
#     print(np.transpose(plot_coord, axes = [1, 2, 3, 0])[np.unravel_index(index, (nx, ny, nz))])