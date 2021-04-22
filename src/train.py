import numpy as np

from agent import Agent
from model import ResidualCNN
from funcs import playMatches

import config

currentNN = ResidualCNN()
bestNN = ResidualCNN()

bestNN.model.set_weights(currentNN.model.get_weights())

currentPlayer = Agent(currentNN)
bestPlayer = Agent(bestNN)
iteration = 0

playMatches(bestPlayer, bestPlayer, 1, config.TURNS_UNTIL_TAU0)
# while True:
#     iteration += 1

#     # TODO: Play games