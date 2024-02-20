import calc
import numpy as np
import config
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

class Animate:
    def __init__(self,grid,ani_every=100,interval=10,frames=10,show=True,scale_cbar=True):
        self.grid = grid
        self.ani_every = ani_every
        self.interval = interval
        self.frames = frames
        self.show = show
        self.scale_cbar = scale_cbar
        self.init_animation()
    def init_animation(self):
        #Get t0 attributes
        self.fig, ax = plt.subplots()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        self.im = ax.imshow(self.grid, cmap='magma', origin='lower',animated=True)#, vmin=cmin, vmax=cmax)
        divider = make_axes_locatable(ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.05)
        self.cbar = self.fig.colorbar(self.im, cax=self.cax)
    def update(self,frame):
        if frame == 0:
            return (self.im,)
        for i in range(self.ani_every):
            self.grid,dgrid = calc.forward_grid(self.grid,config.dx,config.dy,config.dt,config.a
                                                ,method=config.method
                                                ,normalize=config.normalize_grid) 
        self.im.set_data(self.grid)
        #self.im.set_data(dgrid)
        if self.scale_cbar:
            #self.cbar.set_clim(vmin=self.grid.min(), vmax=self.grid.max())
            # Update the normalization of the image and the colorbar
            new_norm = colors.Normalize(vmin=self.grid.min(), vmax=self.grid.max())
            #new_norm = colors.Normalize(vmin=dgrid.min(), vmax=dgrid.max())
            self.im.set_norm(new_norm)
            self.fig.canvas.draw_idle()
        return (self.im,)

    #Animate simulation
    def run_simulation(self):
        ani = animation.FuncAnimation(self.fig, self.update, frames=self.frames, interval=self.interval, blit=True)
        if self.show:
            plt.show()
        return ani

#Simulate t time
def simulate(grid,dx,dy,dt,t,a=1):
    steps = int(t/dt)
    for i in range(steps):
        grid = calc.forward_grid(grid,dx,dy,dt,a)
    return grid