import numpy as np
import random

from MCTS import MCTS, Node, Edge
from game import Game

import config

class Agent:
    def __init__(self, model):
        self.model = model
        self.mcts = None
        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def simulate(self):
        leaf, value, breadcrumbs = self.mcts.moveToLeaf()
        value = self.evaluateLeaf(leaf, value)
        self.mcts.backFill(leaf, value, breadcrumbs)

    def act(self, state, tau):
        if(self.mcts == None or state.toString() not in self.mcts.tree):
            self.buildMCTS(state)
        else:
            self.changeMCTSRoot(state)
        
        for _ in range(config.MCTS_SIMS):
            self.simulate()

        pi, values = self.getAV()

        action, value = self.chooseAction(pi, values, tau)
        
        nextState = state.takeAction(action)

        NN_value= -self.getPredictions(nextState)[0]

        return (action, pi, value, NN_value)

    def getPredictions(self, state):
        inputToModel = np.array([state.toModelInput()])

        preds = self.model.predict(inputToModel)
        value = preds[0][0]
        logits = preds[1][0]

        allowedActions = state.getAvailableMoves()

        mask = np.ones(logits.shape, dtype=bool)
        mask[allowedActions] = False
        logits[mask] = -100

        odds = np.exp(logits)
        probs = odds / np.sum(odds)

        return (value, probs, allowedActions)

    def evaluateLeaf(self, leaf, value):
        if value == 0:
            value, probs, allowedActions = self.getPredictions(leaf.state)
            probs = probs[allowedActions]

            for idx, action in enumerate(allowedActions):
                newState = leaf.state.takeAction(action)
                if newState.toString() not in self.mcts.tree:
                    node = Node(newState)
                    self.mcts.addNode(node)
                else:
                    node = self.mcts.tree[newState.toString()]
                
                newEdge = Edge(leaf, node, probs[idx], action)
                leaf.edges.append(newEdge)
        return value

    def getAV(self):
        edges = self.mcts.root.edges
        pi = np.zeros(384, dtype=np.int)
        values = np.zeros(384, dtype=np.float)

        for edge in edges:
            pi[edge.action] = edge.stats["N"]
            values[edge.action] = edge.stats["Q"]
        
        pi = pi / np.sum(pi)
        return pi, values

    def chooseAction(self, pi, values, tau):
        if tau == 0:
            actions = np.argwhere(pi == max(pi))
            action = random.choice(actions)[0]
        else:
            actionIdx = np.random.multinomial(1, pi)
            action = np.where(actionIdx == 1)[0][0]

        value = values[action]

        return action, value

    def replay(self, memory):
        for _ in range(config.TRAINING_LOOPS):
            batch = random.sample(memory, min(config.BATCH_SIZE, len(memory)))
            inputs = np.array([row["game"].toModelInput() for row in batch])
            targets = {
                "value_head": np.array([row["value"] for row in batch]),
                "policy_head": np.array([row["AV"] for row in batch])
            }
            fit = self.model.model.fit(inputs, targets, epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size=32)
            print("New Loss:")
            print(fit.history)

    def buildMCTS(self, state):
        self.root = Node(state)
        self.mcts = MCTS(Node(state))

    def changeMCTSRoot(self, state):
        self.mcts.root = self.mcts.tree[state.toString()]
