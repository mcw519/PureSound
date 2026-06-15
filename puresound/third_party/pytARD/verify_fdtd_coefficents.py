from common.finite_differences import FiniteDifferences

import matplotlib.pyplot as plt
import numpy as np

'''
Verifies and plots the FDTD first order and laplacian matrix.
'''

plt.close('all')

fig, ax = plt.subplots(ncols=2, nrows=2, sharex=True,
                       sharey=True, figsize=(10, 5))
fig.suptitle('First Order')
for ind, current_ax in enumerate(FiniteDifferences.FD_COEFFICIENTS[1]):
    print(ind)
    current_ax = ax.flatten()[ind]
    curren_fdc = list(FiniteDifferences.FD_COEFFICIENTS[1].values())[ind]
    num_pts = len(curren_fdc)
    current_ax.set_title(f"{num_pts} point stencil")
    current_ax.plot((np.arange(num_pts) - int(num_pts/2)), curren_fdc, '-o')
    current_ax.set_xlabel('dx')
    current_ax.set_ylabel('weight')
    current_ax.set_xlim(-8, 8)
    current_ax.set_ylim(-1, 1)

fig, ax = plt.subplots(ncols=2, nrows=3, sharex=True,
                       sharey=True, figsize=(10, 5))
fig.suptitle('Laplacian')
for ind, current_ax in enumerate(FiniteDifferences.FD_COEFFICIENTS[2]):
    print(ind)
    current_ax = ax.flatten()[ind]
    curren_fdc = list(FiniteDifferences.FD_COEFFICIENTS[2].values())[ind]
    num_pts = len(curren_fdc)
    current_ax.set_title(f"{num_pts} point stencil")
    current_ax.plot((np.arange(num_pts) - int(num_pts/2)), curren_fdc, '-o')
    current_ax.set_xlabel('dx')
    current_ax.set_ylabel('weight')
    current_ax.set_xlim(-8, 8)
    current_ax.set_ylim(min(curren_fdc)-0.5, max(curren_fdc)+0.5)
ax[2, 1].axis('off')

plt.figure(3)
K = FiniteDifferences.get_laplacian_matrix(2, 6)
plt.imshow(K)
plt.show()
print(K)
