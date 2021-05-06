import numpy as np
from collections import deque

import config

class Memory:
    def __init__(self):
        self.longTerm = deque(maxlen=config.MEMORY_SIZE)
        self.shortTerm = deque(maxlen=config.MEMORY_SIZE)

    def commitShortTerm(self, game, actionValues):
        self.shortTerm.append({
            "game": game,
            "AV": actionValues
        })
    
    def commitLongTerm(self):
        for i in self.shortTerm:
            self.longTerm.append(i)
        self.shortTerm = deque(maxlen=config.MEMORY_SIZE)