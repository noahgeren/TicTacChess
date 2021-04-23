from game import Game
from model import ResidualCNN
import constants

import numpy as np


game = Game()
model = ResidualCNN()
preds = model.predict(game.toModelInput())
print(preds[1][0])