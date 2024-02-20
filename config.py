import numpy as np

#set animation params
show = False
folder = 'animations/'
#fname = 'poly1_poly2_wave1_noise1.mp4'
fname = 'test2.mp4'
frames = 100 #number of frames
interval = 100 #interval between frames
ani_every = 1 #number of steps between frames
scale_cbar = True
#set width and height
n = 300 #number of pixels
x,y = 1.,1. #width and height of grid
dt = .1 #time step
a = 1. #diffusion constant
#set scale of pixel
dx,dy = x/n,y/n
use_jit = n>=2000 #use numba if grid is large
#set method
method = 'scipy' #scipy, directional, copy
normalize_grid = True #enforce conservation of energy
#method = 'directional'
#preset grids
cfg_poly1 = {
    #'f':grids.init_polynomial,
    'd':1,
    'b':1,
    'mode':'xy'
}
cfg_poly2 = {
    #'f':grids.init_polynomial,
    'd':2,
    'b':1,
    'mode':'xy'
}
cfg_wave1 = {
    'px':x*(8*np.pi),
    'py':y*(8*np.pi),
    'd':1
}
cfg_noise1 = {
    'amp':1
}
cfg_circle1 = {
    'center':(5,5)
}