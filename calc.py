import numpy as np
import numba as nb
import config
from scipy.ndimage import laplace

use_jit = config.use_jit
# Conditional JIT decorator function
def conditional_jit(func):
    if use_jit:# and config.method != 'scipy':
        return nb.jit(nopython=True)(func)
    else:
        return func

#derivate at single point
@conditional_jit
def derivative(grid,i,dx,axis,order=1,mode='central'):
    #axis 1 is x, axis 0 is y
    if mode == 'central':
        if order == 1:
            if axis == 0:
                return (grid[i+1, :] - grid[i-1, :]) / (2 * dx)
            elif axis == 1:
                return (grid[:, i+1] - grid[:, i-1]) / (2 * dx)
        elif order == 2:
            if axis == 0:
                return (grid[i+1, :] - 2 * grid[i, :] + grid[i-1, :]) / (dx ** 2)
            elif axis == 1:
                return (grid[:, i+1] - 2 * grid[:, i] + grid[:, i-1]) / (dx ** 2)
    elif mode == 'forward':
        if order == 1:
            if axis == 0:
                return (grid[i+1, :] - grid[i, :]) / dx
            elif axis == 1:
                return (grid[:, i+1] - grid[:, i]) / dx
        elif order == 2:
            if axis == 0:
                return (grid[i+2, :] - 2 * grid[i+1, :] + grid[i, :]) / (dx ** 2)
            elif axis == 1:
                return (grid[:, i+2] - 2 * grid[:, i+1] + grid[:, i]) / (dx ** 2)
    elif mode == 'backward':
        if order == 1:
            if axis == 0:
                return (grid[i, :] - grid[i-1, :]) / dx
            elif axis == 1:
                return (grid[:, i] - grid[:, i-1]) / dx
        elif order == 2:
            if axis == 0:
                return (grid[i, :] - 2 * grid[i-1, :] + grid[i-2, :]) / (dx ** 2)
            elif axis == 1:
                return (grid[:, i] - 2 * grid[:, i-1] + grid[:, i-2]) / (dx ** 2)
    else:
        raise ValueError('mode must be central, forward, or backward')


#Solve single time step
@conditional_jit
def forward_grid(grid,dx,dy,dt,a=1,method='scipy',normalize=False):
    #forward euler
    if method == 'scipy':
        dgrid = laplace(grid,mode='nearest')/(dx**2)
        #Regulate edge cases by clipping values from derivative
        gmax = np.max(grid[2:-2,2:-2])
        gmin = np.min(grid[2:-2,2:-2])
        dgrid = np.clip(dgrid,gmin,gmax,out=dgrid)
    else:
        dgrid = differentiate_grid(grid,dx,dy,0,method=method)
        dgrid += differentiate_grid(grid,dx,dy,1,method=method)
    g = grid + dt*a*dgrid
    #scale to integrated temperature of original grid
    if normalize:
        g = g*(np.sum(grid)/np.sum(g))
    return g,dgrid

@conditional_jit
def differentiate_grid(grid,dx,dy,axis,method='directional'):
    #TODO: handle edge cases better?
    dgrid = np.zeros(grid.shape)
    n = grid.shape[0]
    for i in range(2,n-2):
        if axis == 0: #y
            dgrid[i,:] += derivative(grid,i,dy,0,order=2)
        elif axis == 1: #x
            dgrid[:,i] += derivative(grid,i,dx,1,order=2)
    #Regulate edge cases by clipping values from derivative
    gmax = np.max(grid)
    gmin = np.min(grid)
    #handle edge cases
    if method == 'directional':
        #y
        if axis == 0:
            dgrid[0,:] += derivative(grid,0,dy,1,order=2,mode='forward')
            dgrid[1,:] += derivative(grid,1,dy,1,order=2,mode='forward')
            dgrid[-1,:] += derivative(grid,-1,dy,1,order=2,mode='backward')
            dgrid[-2,:] += derivative(grid,-2,dy,1,order=2,mode='backward')
        #x
        if axis == 1:
            dgrid[:,0] += derivative(grid,0,dx,0,order=2,mode='forward')
            dgrid[:,1] += derivative(grid,1,dx,0,order=2,mode='forward')
            dgrid[:,-1] += derivative(grid,-1,dx,0,order=2,mode='backward')
            dgrid[:,-2] += derivative(grid,-2,dx,0,order=2,mode='backward')
    elif method == 'copy':
        if axis == 0:
            dgrid[0,:] = dgrid[2,:]
            dgrid[1,:] = dgrid[2,:]
            dgrid[-1,:] = dgrid[-3,:]
            dgrid[-2,:] = dgrid[-3,:]
        if axis == 1:
            dgrid[:,0] = dgrid[:,2]
            dgrid[:,1] = dgrid[:,2]
            dgrid[:,-1] = dgrid[:,-3]
            dgrid[:,-2] = dgrid[:,-3]
    dgrid = np.clip(dgrid,gmin,gmax,out=dgrid)
    return dgrid

