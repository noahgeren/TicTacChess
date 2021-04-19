from game import Game
from model import Residual_CNN
import constants

import numpy as np


game = Game()
model = Residual_CNN()
preds = model.predict(game.toModelInput())
print(np.reshape(preds[1][0], (24, 4, 4)))