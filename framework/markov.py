# sys
import sys
sys.path.append('code')

# base imports
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# classes and functions
from classes import World, Agent, Surface
from functions import (get_initial_graph,
                       conditional_random_choice,
                       construct_rw_sparse_matrix,
                       sparse_matrix_power)

# initialize world, surface and agents
world = World(200, 200, 200, 20) # 200, 200, 200, 20
surface = Surface(get_initial_graph(world.width, world.length, world.soil_height))
agent_list = [Agent(world) for i in range(500)]

# khuong params
num_steps = 345600 # should be 345600 steps (for 96 hours)
num_agents = 500 # number of agents
m = 8 # num moves per agent

# extra params
collect_data = True
render_images = True
final_render = False
if final_render:
    from render import render

# data storage
pellet_num = 0
total_built_volume = 0
pellet_proportion_list = []
total_surface_area_list = []
total_built_volume_list = []
pickup_rate_list = []
drop_rate_list = []

# start time
start_time = time.time()
# loop over time steps
for step in tqdm(range(num_steps)):
    # reset variables
    removed_indices = []
    # generate randoms for cycle
    random_values = np.random.random(num_agents)
    # create transition matrix and take power
    index_dict, vertices, T = construct_rw_sparse_matrix(surface.graph)
    Tm = sparse_matrix_power(T, m)
    # no pellet num for cycle
    no_pellet_num_cycle, pellet_num_cycle = num_agents-pellet_num, pellet_num
    # pickup and drop rates
    pickup_rate = 0
    drop_rate = 0
    # loop over all agents
    for i in range(num_agents):
        # agent i
        agent = agent_list[i]
        # get position and remove position from index
        prob_dist = Tm[index_dict[agent.pos]].toarray().flatten()
        random_pos = conditional_random_choice(vertices,
                                                p = prob_dist, 
                                                removed_indices=removed_indices)
        agent.pos = random_pos
        has_pellet = agent.has_pellet
        removed_indices.append(index_dict[random_pos])

        # no pellet
        if has_pellet == 0:
            # pickup algorithm
            material = pickup_policy(random_pos, world, x_rand=random_values[i])
            if material is not None:
                # update data and surface
                pellet_num += 1
                pickup_rate += 1/no_pellet_num_cycle
                agent.pos = (random_pos[0],random_pos[1],random_pos[2]-1)
                agent.has_pellet = 1
                if material == 2:
                    total_built_volume -=1
                surface.update_surface(type='pickup', 
                                        pos=random_pos, 
                                        world=world)
                

        # pellet
        else:
            # drop algorithm
            new_pos = drop_policy(random_pos, world, x_rand=random_values[i])
            if new_pos is not None:
                # update data
                pellet_num -= 1
                total_built_volume += 1
                drop_rate += 1/pellet_num_cycle
                agent.pos = new_pos
                agent.has_pellet = 0
                # also remove new position if it is in index dict
                if new_pos in index_dict:
                    removed_indices.append(index_dict[new_pos])
                # update surface
                surface.update_surface(type='drop', 
                                        pos=random_pos, 
                                        world=world)

    # collect data
    if collect_data:
        pellet_proportion_list.append(pellet_num/num_agents)
        total_surface_area_list.append(len(surface.graph.keys()))
        total_built_volume_list.append(total_built_volume)
        pickup_rate_list.append(pickup_rate)
        drop_rate_list.append(drop_rate)

    # if render images
    if render_images:
        if step % (60*60) == 0:
            np.save(file="./exports/tensors/markov_{}".format(step+1), arr=world.grid)

# end time
end_time = time.time()
print("total time taken for this loop: ", end_time - start_time)

# export pandas
if collect_data:
    steps = np.array(range(num_steps))
    params = ['num_steps={}'.format(num_steps),
              'num_agents={}'.format(num_agents),
              'm={}'.format(m),
              'runtime={}s'.format(int(end_time - start_time))]+['']*(num_steps-4)
    data_dict = {
        'params':params,
        'steps':steps,
        'proportion_pellet':pellet_proportion_list,
        'pickup_rate':pickup_rate_list,
        'drop_rate':drop_rate_list,
        'surface_area':total_surface_area_list,
        'volume':total_built_volume_list
    }
    df = pd.DataFrame(data_dict)
    df.to_pickle('./exports/dataframes/markov_khuong_data.pkl')

# render world mayavi
if final_render:
    render(world.grid)