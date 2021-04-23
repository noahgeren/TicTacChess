import numpy as np

from agent import Agent
from model import ResidualCNN
from funcs import playMatches
from memory import Memory

import config

currentNN = ResidualCNN()
bestNN = ResidualCNN()

bestNN.model.set_weights(currentNN.model.get_weights())

currentPlayer = Agent(currentNN)
bestPlayer = Agent(bestNN)

memory = Memory()

iteration = 0

while True:
    iteration += 1
    print("Playing matches...")
    playMatches(bestPlayer, bestPlayer, config.EPISODES, config.TURNS_UNTIL_TAU0, memory)
    memory.clearShortTerm()
    print("Retraining...")
    currentPlayer.replay(memory.longTerm)