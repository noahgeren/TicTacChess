import numpy as np
import random
import sys

from datetime import datetime
from game import Game
from model import ResidualCNN
from agent import Agent

import config

def playMatches(player1, player2, episodes, turnsUntilTau0, memory = None):
    game = Game()
    scores = {
        "player1": 0,
        "player2": 0
    }

    for e in range(episodes):
        print(f'Episode: {e}')
        game.reset()
        winner = 0
        turn = 0
        while winner == 0:
            turn += 1
            
            action, pi, _, _ = (player1 if game.whiteTurn else player2).act(game, 1 if turn < turnsUntilTau0 else 0)

            game = game.takeAction(action)

            # print(np.reshape(game.board, (6,4)))
            # input()
            # This is just in case game logic is wrong and we lose a piece (it has happened before)
            if(sum(1 for i in game.board if i !="-") < 8):
                print(action)
                sys.exit()

            if(memory != None):
                memory.commitShortTerm(game, pi)
            
            winner = game.getWinner()
            if winner != 0:
                if(memory != None):
                    for move in memory.shortTerm:
                        move["value"] = winner if move["game"].whiteTurn != game.whiteTurn else -winner
                    memory.commitLongTerm()
                scores["player1" if winner == 1 else "player2"] += 1
                print(f'Finished in {turn} turns.')
                print(datetime.now())
                print(scores)
            elif turn > 200:
                print("Past 200 turns. No winner.")
                break
    return scores