import numpy as np

#initialize to function f
def initialize_grid(f,n,dx,dy,normalize=True,**kwargs):
    grid = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            grid[i,j] = f(j*dx,i*dy,**kwargs) #inverse order because of the way numpy arrays are indexed
    if normalize: #normalize grid between 0 and 1
        grid = (grid - np.min(grid))/(np.max(grid) - np.min(grid))
    return grid

#Initializer functions
def init_wave(x,y,px,py,d=1.):
    return np.sin(px*x)**d + np.cos(py*y)**d
def init_uniform(x,y):
    return 1
def init_polynomial(x,y,d=2,b=1,mode='xy'):
    val = 0
    if 'x' in mode:
        val += b*x**d
    if 'y' in mode:
        val += b*y**d
    return val
def init_gaussian(x,y,ux,uy,sigma):
    return np.exp(-((x-ux)**2+(y-uy)**2)/(2*sigma**2))
def init_noise(x,y,amp):
    return np.random.rand()*amp
def init_circle(x,y,center):
    #Temperature equals radius from center
    return np.sqrt((x-center[0])**2 + (y-center[1])**2)

#Other grid fill functions
def fill_semi_circle(grid, center, radius):
    h, k = center
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            # formula for a circle (x-h)^2 + (y-k)^2 = r^2
            if ((x - h)**2) + ((y - k)**2) <= radius**2 and y >= k:
                grid[y,x] = 1
    return grid
def generate_rocket_grid(grid_size):
    grid = np.ones((grid_size, grid_size)) * -1  # Start with a black grid

    # Draw the rocket as white cells on the grid
    mid_point = grid_size // 2
    #grid[grid_size - mid_point:grid_size, mid_point] = 1 # rocket body
    grid[grid_size - mid_point * 2:grid_size - mid_point, mid_point - int(grid_size/6):mid_point + int(grid_size/6)] = 1 # rocket body

    # Draw the rocket head as a semi circle
    grid = fill_semi_circle(grid, (mid_point, mid_point), (grid_size - mid_point)/2.5)

    return grid

