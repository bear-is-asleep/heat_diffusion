import animate
import grids
import config
import numpy as np
import os

#Initialize grid and simulation
n = config.n
dx,dy = config.dx,config.dy
dt = config.dt
a = config.a

#Which grids do we want?
grid_v = [
    #grids.initialize_grid(grids.init_polynomial,n,dx,dy,**config.cfg_poly1),
    grids.initialize_grid(grids.init_polynomial,n,dx,dy,**config.cfg_poly2),
    grids.initialize_grid(grids.init_wave,n,dx,dy,**config.cfg_wave1),
    #0.001*grids.initialize_grid(grids.init_noise,n,dx,dy,normalize=False,**config.cfg_noise1),
    #grids.initialize_grid(grids.init_circle,n,dx,dy,**config.cfg_circle1)
]

grid = np.sum(grid_v,axis=0)


#Run animator
folder = config.folder
fname = config.fname
frames = config.frames
interval = config.interval
ani_every = config.ani_every
show = config.show
scale_cbar = config.scale_cbar

Ani = animate.Animate(grid,ani_every=ani_every,interval=interval,frames=frames,show=show,scale_cbar=scale_cbar)
Ani.init_animation()
ani = Ani.run_simulation()
if not config.show:
    os.makedirs(config.folder, exist_ok=True)
    ani.save(folder+fname, writer='ffmpeg', fps=30)
