# sys
import sys
sys.path.append('code')

# base imports
import numpy as np
import pandas as pd
import time

# classes and functions
from classes import World
from display import render

# initialize world, object and tensor to diffuse
obj = np.zeros((60, 60, 60))
obj[23:27, 23:27, 20:40] = 1
tensor = np.zeros((60, 60, 60))
tensor[30, 30, 21] = 100
world = World(60, 60, 60, 20, objects = [obj]) # 200, 200, 200, 20

# start time
start_time = time.time()
# diffuse
new_tensor = world.diffuse(tensor, diffusion_rate=0.25, num_iterations=1)
# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)

# print tensor within object
print(new_tensor)

# plot
world.grid[np.where(new_tensor!=0)]=2
render(world.grid)

'''
Seems diffusion is working as we would expect, bouncing against objects
and leaking out of the world on boundaries. It takes about 1 second per
iteration in a 60*60*60 world.
'''