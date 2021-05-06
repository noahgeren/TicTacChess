import numpy as np
import config

class Node:
    def __init__(self, state):
        self.state = state
        self.edges = []
        self.id = state.toString()
    
    def isLeaf(self):
        return len(self.edges) == 0

class Edge:
    def __init__(self, inNode, outNode, prior, action):
        self.inNode = inNode
        self.outNode = outNode
        self.action = action

        self.stats = {
            "N": 0,
            "W": 0,
            "Q": 0,
            "P": prior
        }

class MCTS:
    def __init__(self, root):
        self.root = root
        self.tree = {}
        self.addNode(root)

    def moveToLeaf(self):
        breadcrumbs = []
        currentNode = self.root
        value = 0
        depth = 0
        while not currentNode.isLeaf():
            maxQU = -999999999
            if currentNode == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(currentNode.edges))
            else:
                epsilon = 0
                nu = [0] * len(currentNode.edges)
            Nb = 0
            for edge in currentNode.edges:
                Nb += edge.stats["N"]
            for idx, edge in enumerate(currentNode.edges):
                if(edge in breadcrumbs): continue
                U = config.CPUCT * ((1-epsilon) * edge.stats["P"] + epsilon * nu[idx]) * np.sqrt(Nb) / (1 + edge.stats["N"])
                Q = edge.stats["Q"]
                if Q + U > maxQU:
                    maxQU = Q + U
                    simulationEdge = edge
            value = (1 if currentNode.state.whiteTurn else -1) * currentNode.state.takeAction(simulationEdge.action).getWinner()
            currentNode = simulationEdge.outNode
            breadcrumbs.append(simulationEdge)
            depth += 1
            if(depth > 100): break
        return (currentNode, value, breadcrumbs)

    def backFill(self, leaf, value, breadcrumbs):
        currentPlayer = leaf.state.whiteTurn
        for edge in breadcrumbs:
            playerTurn = edge.inNode.state.whiteTurn
            direction = 1 if playerTurn != currentPlayer else -1
            edge.stats["N"] += 1
            edge.stats["W"] += value * direction
            edge.stats["Q"] = edge.stats["W"] / edge.stats["N"]

    def addNode(self, node):
        self.tree[node.id] = node