# sys
import sys
sys.path.append('code')

# base imports
import numpy as np
import pandas as pd
import time

# classes and functions
from classes import World, Agent, Surface, Structure
from functions import get_initial_graph
from display import render

# testing the classes
world = World(60, 60, 60, 10)
agent = Agent(world)
surface = Surface(get_initial_graph(world.width, world.length, world.soil_height))
structure = Structure()
