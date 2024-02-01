# ------------ Imports----------------------
# ------------------------------------------

import numpy as np
import scipy.sparse as sp
import torch

# ------------ Useful Lists ----------------
# ------------------------------------------

translation_dict = {-2:'agent', -1:'object', 0:'void', 1:'soil',2:'pellet'}
neighbour_directions_3d = np.array([
    [1, 0, 0],   # Right
    [-1, 0, 0],  # Left
    [0, 1, 0],   # Forward
    [0, -1, 0],  # Backward
    [0, 0, 1],   # Up
    [0, 0, -1],  # Down
    [1, 1, 0],   # Right-Forward
    [1, -1, 0],  # Right-Backward
    [-1, 1, 0],  # Left-Forward
    [-1, -1, 0], # Left-Backward
    [1, 0, 1],   # Right-Up
    [1, 0, -1],  # Right-Down
    [-1, 0, 1],  # Left-Up
    [-1, 0, -1], # Left-Down
    [0, 1, 1],   # Forward-Up
    [0, 1, -1],  # Forward-Down
    [0, -1, 1],  # Backward-Up
    [0, -1, -1], # Backward-Down
    [1, 1, 1],   # Right-Forward-Up
    [1, 1, -1],  # Right-Forward-Down
    [1, -1, 1],  # Right-Backward-Up
    [1, -1, -1], # Right-Backward-Down
    [-1, 1, 1],  # Left-Forward-Up
    [-1, 1, -1],  # Left-Forward-Down
    [-1, -1, 1],  # Left-Backward-Up
    [-1, -1, -1] # Left-Backward-Down
])
neighbour_directions = np.array([
    [0, 0, 1], # Up
    [0, 0, -1], # Down
    [0, 1, 0], # Forward
    [0, -1, 0], # Backwards
    [1, 0, 0], # Right
    [-1, 0, 0] # Left
])
neighbour_filter = np.array([
       [[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]],
       [[0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]],
       [[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]]])
center_loc = np.array([1,1,1])

# ------------ Helper functions ----------------
# ----------------------------------------------

# random initial config
def random_initial_config(width, lenght, soil_height, num_agents, objects=None):
    """
    Generates a random initial configuration for agents in a given space.

    Parameters:
    - width: Width of the space.
    - length: Length of the space.
    - soil_height: Height of the soil.
    - num_agents: Number of agents.
    - objects: Objects in world (To Implement)

    Returns:
    - OrderedDict with agent configurations.
    """
    open_initial_positions = []
    agent_dict = {} # not ordered
    for i in range(width):
        for j in range(lenght):
            open_initial_positions.append((i,j,soil_height))
    
    random_positions = random_choices(open_initial_positions, size=num_agents, p=None)
    for i,pos in enumerate(random_positions):
        agent_dict[i] = [pos,0]
    return agent_dict

# get initial surface graph
def get_initial_graph(width, lenght, soil_height):
    """
    Generates an initial surface graph for a given world.

    Parameters:
    - width: Width of the space.
    - length: Length of the space.
    - soil_height: Height of the soil.

    Returns:
    - Initial surface graph.
    """
    # initialize
    graph = {}
    # inside cases
    for i in range(1,width-1):
        for j in range(1,lenght-1):
            graph[(i,j,soil_height)] = [(i-1,j,soil_height),(i,j-1,soil_height),(i+1,j,soil_height),(i,j+1,soil_height)]
    # sides
    for j in range(1,lenght-1):
        graph[(0,j,soil_height)] = [(0,j-1,soil_height),(1,j,soil_height),(0,j+1,soil_height)]
        graph[(width-1,j,soil_height)] = [(width-1,j-1,soil_height),(width-2,j,soil_height),(width-1,j+1,soil_height)]
    # sides
    for i in range(1,width-1):
        graph[(i,0,soil_height)] = [(i-1,0,soil_height),(i,1,soil_height),(i+1,0,soil_height)]
        graph[(i,lenght-1,soil_height)] = [(i-1,lenght-1,soil_height),(i,lenght-2,soil_height),(i+1,lenght-1,soil_height)]
    # corners
    graph[(0,0,soil_height)] = [(1,0,soil_height),(0,1,soil_height)]
    graph[(width-1,lenght-1,soil_height)] = [(width-2,lenght-1,soil_height),(width-1,lenght-2,soil_height)]
    graph[(width-1,0,soil_height)] = [(width-2,0,soil_height),(width-1,1,soil_height)]
    graph[(0,lenght-1,soil_height)] = [(0,lenght-2,soil_height),(1,lenght-1,soil_height)]
    # return
    return graph

# random scatter of material
def random_scatter(num_scatter, world, surface, structure):
    """
    Randomly scatter a specified number of material drops on the given world.

    Parameters:
    - num_scatter (int): Number of material drops to scatter.
    - world (World): The world object representing the environment.
    - surface (Surface): The surface object for updating surface information.
    - structure (Structure): The structure object for updating structure information.

    Returns:
    None

    Description:
    This function randomly scatters material drops within the world. It iterates 'num_scatter' times,
    attempting to drop material at random positions. If the selected position is occupied, it searches
    vertically until it finds an empty space to place the material drop. The changes in the world, surface,
    and structure are updated accordingly.

    Note:
    The world grid is modified in-place, and the surface and structure are updated with the type 'drop'
    at the corresponding positions.

    Example:
    random_scatter(10, my_world, my_surface, my_structure)
    """
    for i in range(num_scatter):
        random_pos = (np.random.randint(world.width), np.random.randint(world.length), world.soil_height)
        if world.grid[random_pos[0],random_pos[1],random_pos[2]] == 0:
            world.grid[random_pos[0],random_pos[1],random_pos[2]] = 2
            surface.update_surface(type='drop', 
                                        pos=random_pos, 
                                        world=world)
            structure.update_structure(type='drop', 
                                        pos=random_pos, 
                                        material=None)
        else:
            j = 0
            while True:
                j += 1
                if world.grid[random_pos[0],random_pos[1],random_pos[2]+j] == 0:
                    new_pos = (random_pos[0],random_pos[1],random_pos[2]+j)
                    world.grid[new_pos[0],new_pos[1],new_pos[2]] = 2
                    surface.update_surface(type='drop', 
                                                pos=new_pos, 
                                                world=world)
                    structure.update_structure(type='drop', 
                                                pos=new_pos, 
                                                material=None)
                    break

# get local grid data from position in world
def local_grid_data(pos, world):
    """
    Retrieves local grid data from a position in the world.
    Does not seem to be optimizable.

    Parameters:
    - pos: Position in the world.
    - world: The world object.

    Returns:
    - Local grid data.
    """
    x,y,z = pos
    if 0<x<world.width-2 and 0<y<world.length-2 and 0<z<world.height-2:
        # Not on any boundary -- No padding
        local_data = world.grid[x-1:x+2,y-1:y+2,z-1:z+2]
        return local_data
    else:
        # On some boundary -- Need padding
        x_min, x_max = max(0, x - 1), min(world.width, x + 2)
        y_min, y_max = max(0, y - 1), min(world.length, y + 2)
        z_min, z_max = max(0, z - 1), min(world.height, z + 2)
        # slicing start and end indices
        x_start, x_end = 1 if x_min == 0 else 0, 2 if x_max == world.width else 3
        y_start, y_end = 1 if y_min == 0 else 0, 2 if y_max == world.length else 3
        z_start, z_end = 1 if z_min == 0 else 0, 2 if z_max == world.height else 3
        # initialize with -1 then fill with available data
        local_data = np.full((3, 3, 3), -1)
        local_data[x_start:x_end, y_start:y_end, z_start:z_end] = world.grid[
            x_min:x_min + x_end - x_start, y_min:y_min + y_end - y_start, z_min:z_min + z_end - z_start]
        # return
        return local_data

# a height function that returns the number of empty cells below agent
def compute_height(pos, world):
    """
    Computes the height (number of empty cells below) for a given position in the world.
    Does not seem to be optimizable.

    Parameters:
    - pos: Position in the world.
    - world: The world object.

    Returns:
    - Number of empty cells below the agent.
    """
    x,y,z = pos
    max_h = world.height
    h=0
    for i in range(max_h):
        z-=1
        cell = world.grid[x,y,z]
        if cell > 0:
            return h
        h+=1
    return max_h

# returns the valid move directions from local data
def valid_move_directions(local_data):
    """
    Returns valid move directions based on local data.
    Optimizable in the sense that we could return as we search randomly.

    Parameters:
    - local_data: Local data tensor.

    Returns:
    - List of valid move directions.
    """
    dirs = []
    # loop over 6 neighbours
    for dir in neighbour_directions:
        new_pos = center_loc + dir
        if local_data[new_pos[0], new_pos[1], new_pos[2]]==0:
            # slice lower bounds
            x_low_bound = max(0, new_pos[0]-1)
            y_low_bound = max(0, new_pos[1]-1)
            z_low_bound = max(0, new_pos[2]-1)
            # sliced array
            new_local_data = local_data[x_low_bound:new_pos[0]+2, y_low_bound:new_pos[1]+2, z_low_bound:new_pos[2]+2]
            # check for any material
            if (new_local_data>0).any():
                dirs.append(dir)
    return dirs

# returns a random move direction from local data or None
def random_move_direction(local_data):
    """
    Returns valid move directions based on local data.
    Runs faster within code because we return as soon as we find.

    Parameters:
    - local_data: Local data tensor.

    Returns:
    - List of valid move directions.
    """
    # permutation for moving
    permutation = np.random.permutation(6)
    # loop over 6 neighbours
    for i in permutation:
        dir = neighbour_directions[i]
        new_pos = center_loc + dir
        if local_data[new_pos[0], new_pos[1], new_pos[2]]==0:
            # slice lower bounds
            x_low_bound = max(0, new_pos[0]-1)
            y_low_bound = max(0, new_pos[1]-1)
            z_low_bound = max(0, new_pos[2]-1)
            # sliced array
            new_local_data = local_data[x_low_bound:new_pos[0]+2, y_low_bound:new_pos[1]+2, z_low_bound:new_pos[2]+2]
            # check for any material
            if (new_local_data>0).any():
                return dir
    return None

# checks neighbours for some material
def voxel_shares_face_with_material(local_data):
    """
    Checks if neighbors share a face with some material.
    Does not seem to be optimizable.

    Parameters:
    - local_data: Local data tensor.

    Returns:
    - Binary condition indicating if neighbors share a face with material.
    """
    filtered = local_data*neighbour_filter
    shares_face = (filtered > 0).any()
    # return binary condition
    return shares_face

# list of valid actions from local data tensor
def valid_actions(local_data, has_pellet):
    """
    Returns a list of valid actions based on local data and whether the agent has a pellet.
    I dont use this much anymore.

    Parameters:
    - local_data: Local data tensor.
    - has_pellet: Whether the agent has a pellet.

    Returns:
    - List of valid actions.
    """
    action_list = []
    move_directions = valid_move_directions(local_data)
    # move
    if len(move_directions)>0:
        for dir in move_directions:
            action_list.append(('move',dir)) # movements
        # drop
        if has_pellet is True:
            if voxel_shares_face_with_material(local_data):
                ind = np.random.randint(0,len(move_directions))
                dir = move_directions[ind]
                action_list.append(('drop',dir)) # drop with random movement
    # pickup
    if has_pellet is False:
        under = np.array([0,0,-1])
        if local_data[1,1,0] in [1,2]:
            action_list.append(('pickup',under)) # pickup

    return action_list

# list of valid moves from local data tensor
def valid_moves(local_data):
    """
    Returns a list of valid move actions based on local data.
    I dont use this much anymore.

    Parameters:
    - local_data: Local data tensor.

    Returns:
    - List of valid move actions.
    """
    action_list = []
    move_directions = valid_move_directions(local_data)
    # moves in format
    if len(move_directions)>0:
        for dir in move_directions:
            action_list.append(('move',dir)) # movements
    return action_list

# get neighbours of some voxel given current graph
def get_neighbours(pos, graph):
    """
    Retrieves neighbors of a voxel given the current graph.
    Does not seem to be optimizable.

    Parameters:
    - pos: Position in the world.
    - graph: Current graph.

    Returns:
    - List of neighbors.
    """
    neighbours = []
    # loop over potential neighbours
    for dir in neighbour_directions:
        new_pos = (pos[0]+dir[0],pos[1]+dir[1],pos[2]+dir[2])
        # check if is in graph
        if new_pos in graph:
            # if yes, append
            neighbours.append(new_pos)
    return neighbours

# update surface graph
def update_surface_function(surface, type, pos, world):
    """
    Updates the surface graph based on a specific action type.
    Does not seem to be optimizable.

    Parameters:
    - surface: Surface object.
    - type: Type of action ('pickup' or 'drop').
    - pos: Position in the world.
    - world: The world object.
    """
    # position and tuple version
    x,y,z = pos
    vertex = (x,y,z)
    # case 1
    if type == 'pickup':
        new_vertex = (x,y,z-1)
        # add new vertex to graph and degrees
        surface.graph[new_vertex] = []
        surface.degrees[new_vertex] = 0
        # get neighbours
        nbr_list = get_neighbours(new_vertex,surface.graph)
        for nbr in nbr_list:
            # this adds edge to both and also updates degrees
            surface.add_edge((new_vertex, nbr))
    # case 2
    elif type == 'drop':
        # remove vertex at location of drop
        surface.remove_vertex(vertex)
        # initialize new vertices
        new_vertices = []
        # loop over 3D directions
        for dir in neighbour_directions_3d:
            # new vertex position
            new_vertex = (pos[0]+dir[0],pos[1]+dir[1],pos[2]+dir[2])
            # check in bounds
            if 0<=new_vertex[0]<world.width and 0<=new_vertex[1]<world.length and 0<=new_vertex[2]<world.height:
                # check empty
                if world.grid[new_vertex[0],new_vertex[1],new_vertex[2]]==0:
                    # check it isnt already in graph
                    if new_vertex not in surface.graph:
                        # accept as new vertex in surface
                        new_vertices.append(new_vertex)
        # loop over newly added surface vertices
        for new_vertex in new_vertices:
            # add to graph and degrees dicts
            surface.graph[new_vertex] = []
            surface.degrees[new_vertex] = 0
            # get neighbours (with current graph)
            nbr_list = get_neighbours(new_vertex,surface.graph)
            # loop over neighbours
            for nbr in nbr_list:
                # add new edges (exactly once because of sequentiality)
                surface.add_edge((new_vertex, nbr))
    # case 3
    else:
        raise ValueError('type not in pickup, drop')

# update structure graph
def update_structure_function(structure, type, pos, material=None):
    """
    Updates the structure graph based on a specific action type.
    Does not seem to be optimizable.

    Parameters:
    - structure: Surface object.
    - type: Type of action ('pickup' or 'drop').
    - pos: Position in the world of agent.
    - material: The picked up material if any
    """
    # position and vertex
    x,y,z = pos
    vertex = (x,y,z)

    # case 1
    if type == 'pickup':
        new_vertex = (x,y,z-1)
        # check if material is 2
        if material == 2:
            # remove vertex including all it's edges
            structure.remove_vertex(new_vertex)

    # case 2
    elif type == 'drop':
        # add vertex to graph and connect it accordingly
        structure.graph[vertex] = []
        # loop over neighbour directions
        for dir in neighbour_directions:
            # the neighbour (supposedly)
            new_vertex = (x+dir[0],y+dir[1],z+dir[2])
            # check if it is in structure graph
            if new_vertex in structure.graph:
                # if so, add the edge
                structure.add_edge((vertex, new_vertex))

    # case 3
    else:
        raise ValueError('type not in pickup, drop')

# ------------ Random Sampling ---------------
# --------------------------------------------

# random choices function
def random_choices(ls, size=1, p=None):
    """
    Performs random choices from a list with or without given probabilities.
    Does not seem to be optimizable.

    Parameters:
    - ls: List of elements.
    - size: Number of elements to choose.
    - p: Probabilities.

    Returns:
    - List of randomly chosen elements.
    """
    # No prob dist given
    if p is None:
        chosen_indices = np.random.choice(range(len(ls)), size, replace=False)
        choices = [ls[i] for i in chosen_indices]
        return choices
    # A prob dist is given
    else:
        chosen_indices = np.random.choice(range(len(ls)), size, p=p, replace=False)
        choices = [ls[i] for i in chosen_indices]
        return choices
    
# conditional random choices function
def conditional_random_choice(ls, p=None, removed_indices=None):
    """
    Performs a random choice with a condition on the probabilities.
    Does not seem to be optimizable.

    Parameters:
    - ls: List of elements (tuples).
    - p: Probabilities.
    - removed_indices: Indices to be removed from consideration.

    Returns:
    - A randomly chosen element.
    """
    # No prob dist given
    if p is None:
        return ls[np.random.choice(len(ls))]
    # A prob dist is given
    else:
        new_p = p.copy()
        new_p[removed_indices] = 0
        new_p /= np.sum(new_p)
        chosen_index = np.random.choice(len(ls), p=new_p)
        return ls[chosen_index]
    
# dual random choices function
def dual_random_choices(ls, size0, size1, prob0, prob1=None):
    """
    Performs random choices from two distributions.
    I dont use this anymore.

    Parameters:
    - ls: List of elements.
    - size0: Number of elements to choose from the first distribution.
    - size1: Number of elements to choose from the second distribution.
    - prob0: Probabilities for the first distribution.
    - prob1: Probabilities for the second distribution.

    Returns:
    - Two lists of randomly chosen elements.
    """
    # index choices
    num_elements = len(ls)
    choices0 = []
    choices1 = []
    # convert to arrays
    prob0 = np.nan_to_num(prob0.toarray().flatten())
    prob0 = prob0/np.sum(prob0)
    if prob1.count_nonzero()==0:
        prob1 = None
    else:
        prob1 = np.nan_to_num(prob1.toarray().flatten())
        prob1 /= np.sum(prob1)
    # Generate choices from the first distribution
    chosen_indices0 = np.random.choice(range(num_elements), size0, p=prob0, replace=False)
    choices0 = [ls[i] for i in chosen_indices0]
    # Generate choices from the second distribution
    remaining_indices = np.setdiff1d(range(num_elements), chosen_indices0)  
    # Normalize prob1 over the remaining indices
    if prob1 is not None:
        summed = np.sum(prob1[remaining_indices])
        if summed > 0:
            prob1_normalized = prob1[remaining_indices] / np.sum(prob1[remaining_indices])
        else:
            prob1_normalized = None
    else:
        prob1_normalized = None
    chosen_indices1 = np.random.choice(remaining_indices, size1, p=prob1_normalized, replace=False)
    choices1 = [ls[i] for i in chosen_indices1]
    # return
    return choices0, choices1

# ------------ Linear Algebra ----------------
# --------------------------------------------

# construct sparse tensors
def construct_rw_sparse_matrix(graph):
    """
    Constructs a sparse matrix representation for a given graph.
    This is expensive, but I dont know if I can optimize it.

    Parameters:
    - graph: Graph representing the world.

    Returns:
    - Index dictionary, vertices, and sparse matrix.
    """
    # variables
    vertices = list(graph.keys())
    num_vertices = len(vertices)
    
    # build index
    index_dict = {vertex: i for i, vertex in enumerate(vertices)}
    
    # initialize coo_matrix
    T = sp.coo_matrix((num_vertices, num_vertices))

    # build T
    row_indices = []
    col_indices = []
    data = []
    for i, vertex in enumerate(vertices):
        nbrs = graph[vertex]
        num_nbrs = len(nbrs)
        map_nbrs = [index_dict[v] for v in nbrs]   
        if num_nbrs>0:
            row_indices.extend([i] * num_nbrs)
            col_indices.extend(map_nbrs)
            data.extend([1 / num_nbrs] * num_nbrs)
        else:
            row_indices.extend([i])
            col_indices.extend([i])
            data.extend([1 / num_nbrs] * num_nbrs)

    # build T
    T = sp.coo_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices)).tocsr()

    # return
    return index_dict, vertices, T

# a function that takes power of matrix through repeated squaring
def sparse_matrix_power(A, m):
    """
    Computes the power of a sparse matrix using repeated squaring.
    This is expensive, but I dont know if I can optimize it.

    Parameters:
    - A: Sparse matrix.
    - m: Exponent.

    Returns:
    - Resulting sparse matrix.
    """
    # Initialize with the identity matrix
    result = sp.eye(A.shape[0], format='csr')
    # Use repeated squaring to calculate A^m
    while m > 0:
        if m % 2 == 1:
            result = result @ A
        A = A @ A
        m //= 2
    # return
    return result