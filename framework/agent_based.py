# sys
import sys
sys.path.append('code')

# base imports
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

# classes
from classes import World, Agent

# initialize world and agents
world = World(200, 200, 200, 20) # 200, 200, 200, 20
agent_list = [Agent(world) for i in range(500)]

# params
num_steps = 100 
num_agents = 500 
m = 15

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
total_built_volume_list = []
pickup_rate_list = []
drop_rate_list = []

# start time
start_time = time.time()
# loop over time steps
for step in tqdm(range(num_steps)):
    # reset variables and generate randoms
    x_random = np.random.rand(num_agents)
    # no pellet num for cycle
    no_pellet_num_cycle, pellet_num_cycle = num_agents-pellet_num, pellet_num
    # pickup and drop rates
    pickup_rate = 0
    drop_rate = 0
    # loop over agents
    for i in range(num_agents):
        # agent
        agent = agent_list[i]

        # movement rule
        movement_policy(agent, world, m)

        # pickup algorithm
        if agent.has_pellet == 0:
            material = pickup_policy(agent.pos, world, x_rand=x_random[i])
            if material is not None:
                # pickup
                agent.pickup(world)
                # data updates
                pellet_num += 1
                pickup_rate += 1/no_pellet_num_cycle
                if material == 2:
                    total_built_volume -=1

        # drop algorithm
        else:
            new_pos = drop_policy(agent.pos, world, x_rand=x_random[i])
            if new_pos is not None:
                # drop
                agent.drop(world, new_pos)
                # data updates
                pellet_num -= 1
                drop_rate += 1/pellet_num_cycle
                total_built_volume += 1

    # collect data
    if collect_data:
        pellet_proportion_list.append(pellet_num/num_agents)
        total_built_volume_list.append(total_built_volume)
        pickup_rate_list.append(pickup_rate)
        drop_rate_list.append(drop_rate)

    # render images
    if render_images:
        if step % (60*60) == 0:
            np.save(file="./exports/tensors/original_{}".format(step+1), arr=world.grid)

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
        'volume':total_built_volume_list
    }
    df = pd.DataFrame(data_dict)
    df.to_pickle('./exports/dataframes/agent_based_data.pkl')

# render world mayavi
if final_render:
    render(world.grid)