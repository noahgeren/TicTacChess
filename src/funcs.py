import numpy as np
import random

from game import Game
from model import ResidualCNN
from agent import Agent

import config

def playMatches(player1, player2, episodes, turnsUntilTau0, memory = None, goesFirst = 0):
    game = Game()
    scores = {
        "player1": 0,
        "player2": 0,
        "stalemate": 0
    }

    for e in range(episodes):
        print(f'Episode: {e}')
        game.reset()

        player1Starts = goesFirst
        if goesFirst == 0:
            player1Starts = random.randint(0, 1) * 2 - 1
        if player1Starts == 1:
            players = {
                1: {
                    "agent": player1,
                    "name": "player1"
                },
                -1: {
                    "agent": player2,
                    "name": "player2"
                }
            }
        else:
            players = {
                1: {
                    "agent": player2,
                    "name": "player2"
                },
                -1: {
                    "agent": player1,
                    "name": "player1"
                }
            }
        winner = 0
        turn = 0
        while winner == 0:
            turn += 1
            print(f'Turn: {turn}')
            print(np.reshape(game.board, (6, 4)))
            action, pi, _, _ = players[1 if game.whiteTurn else -1]["agent"].act(game, 1 if turn < turnsUntilTau0 else 0)

            game = game.takeAction(action)

            if(memory != None):
                memory.commitShortTerm(game, pi)
            
            winner = game.getWinner()
            if winner != 0:
                if(memory != None):
                    for move in memory.shortTerm:
                        move["value"] = winner if move["game"].whiteTurn == game.whiteTurn else -winner
                    memory.commitLongTerm()
                if(winner == 1):
                    scores[players[1 if game.whiteTurn else -1]["name"]] += 1
                else:
                    scores[players[-1 if game.whiteTurn else 1]["name"]] += 1
    return scores