import numpy as np
from collections import deque

import config

class Memory:
    def __init__(self):
        self.longTerm = deque(maxlen=config.MEMORY_SIZE)
        self.shortTerm = deque(maxlen=config.MEMORY_SIZE)

    def commitShortTerm(self, game, actionValues):
        for i in game.identities(actionValues):
            self.shortTerm.append({
                "game": i[0],
                "AV": i[1]
            })
    
    def commitLongTerm(self):
        for i in self.shortTerm:
            self.longTerm.append(i)
        self.clearShortTerm()

    def clearShortTerm(self):
        self.shortTerm = deque(maxlen=config.MEMORY_SIZE)