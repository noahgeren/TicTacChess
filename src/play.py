# from game import Game
# from model import ResidualCNN

# import numpy as np


# game = Game()
# model = ResidualCNN()
# preds = model.predict(game.toModelInput())
# print(preds[1][0])

# import pygame
# from pygame.locals import *
# import sys
# pygame.init()
# size = width, height = 400, 400
# screen = pygame.display.set_mode(size)

# while True:
#     for event in pygame.event.get():
#         if event.type == QUIT:
#             sys.exit()

from game import Game

game = Game("PNBR----------------rbnpbfb1")

print(game.toModelInput())
    
