import numpy as np
import pickle
import os

from agent import Agent
from model import ResidualCNN
from funcs import playMatches
from memory import Memory

import config

# This can be set to load existing data
initialNN = None
initialMemory = None

currentNN = ResidualCNN()
bestNN = ResidualCNN()

bestPlayerVersion = 0

if initialNN == None:
    bestNN.model.set_weights(currentNN.model.get_weights())
else:
    bestPlayerVersion = initialNN
    print(f'Loading NN version {bestPlayerVersion}...')
    bestNN.read(f'../models/version{initialNN}.h5')
    currentNN.model.set_weights(bestNN.model.get_weights())

currentPlayer = Agent(currentNN)
bestPlayer = Agent(bestNN)

os.makedirs(os.path.dirname(f'../memory/version0.p'), exist_ok=True)
if initialMemory == None:
    memory = Memory()
else:
    print(f'Loading memory version {initialMemory}...')
    memory = pickle.load(open(f'../memory/version{initialMemory}.p', "rb"))

iteration = 0

while True:
    iteration += 1
    print(f'Iteration {iteration}:')
    print("Playing matches...")
    playMatches(bestPlayer, bestPlayer, config.EPISODES, config.TURNS_UNTIL_TAU0, memory)

    pickle.dump(memory, open(f'../memory/version{iteration}.p', "wb"))

    if(len(memory.longTerm) >= config.MEMORY_SIZE):
        print("Retraining...")
        currentPlayer.replay(memory.longTerm)
        
        print("Tournament...")
        scores = playMatches(currentPlayer, bestPlayer, config.EVAL_EPISODES, 0, memory)
        print("Scores:")
        print(scores)
        if(scores["player1"] > scores["player2"] * config.SCORING_THRESHOLD):
            bestPlayerVersion += 1
            bestNN.model.set_weights(currentNN.model.get_weights())
            bestNN.write(f'../models/version{bestPlayerVersion}.h5')