import constants
import numpy as np

class Game:

    defaultGameState = "----------------PNBRrbnpffw0"

    def getWinner(self): # 1 for white, -1 for black, 0 for no winner
        for win in constants.wins:
            if("-" in [self.board[i] for i in win]):
                continue
            winner = constants.pieces[self.board[win[0]]]["color"]
            if([constants.pieces[self.board[i]]["color"] for i in win].count(winner) == 4):
                return  1 if winner == "white" else -1
        return 0

    def getAvailableMoves(self):
        allowed = []
        startIndex = 0 if self.afterThreeMoves else 16
        for fromIndex in range(startIndex, 24):
            if self.board[fromIndex].isupper() == self.whiteTurn:
                available = self.__getAvailableMovesFromPos(fromIndex)
                for toIndex in available:
                    allowed.append(fromIndex * 16 + toIndex)
        return allowed

    def toString(self):
        return Game.stateToString(self.board, self.whiteForward, self.blackForward, self.whiteTurn, self.afterThreeMoves)

    def toModelInput(self):
        modelInput = []
        # piece planes
        for piece in constants.pieces:
            plane = np.zeros(24, dtype=np.int)
            plane[self.board==(piece if self.whiteTurn else piece.swapcase())] = 1
            modelInput = np.append(modelInput, plane)
        # pawn directions
        whiteForward = np.ones(24, dtype=np.int) if self.whiteForward else np.zeros(24, dtype=np.int)
        blackForward = np.ones(24, dtype=np.int) if self.blackForward else np.zeros(24, dtype=np.int)
        if(self.whiteTurn):
            modelInput = np.append(modelInput, whiteForward)
            modelInput = np.append(modelInput, blackForward)
        else:
            modelInput = np.append(modelInput, blackForward)
            modelInput = np.append(modelInput, whiteForward)
        # after three moves
        modelInput = np.append(modelInput, np.ones(24, dtype=np.int) if self.afterThreeMoves else np.zeros(24, dtype=np.int))
        return np.reshape(modelInput, (11, 6, 4))
    
    # Note: This method does not check if an action is legal
    def takeAction(self, action): # action is 0-384
        fromIndex = action // 16
        toIndex = action % 16
        newBoard = np.array(self.board)
        to = newBoard[toIndex]
        piece = newBoard[fromIndex]
        if(to != "-"): # Replace piece to offboard
            newBoard[16 + list(constants.pieces).index(to)] = to
        newBoard[toIndex] = piece
        newBoard[fromIndex] = "-"
        whiteForward = self.whiteForward
        blackForward = self.blackForward
        if(piece.lower() == "p" and toIndex in [0, 1, 2, 3, 12, 13, 14, 15]):
            if(piece.isupper()):
                whiteForward = not whiteForward
            else:
                blackForward = not blackForward
        afterThreeMoves = self.afterThreeMoves
        if(not afterThreeMoves and np.count_nonzero(newBoard[:16] != "-") <= 10):
            afterThreeMoves = True
        return Game(Game.stateToString(newBoard, whiteForward, blackForward, not self.whiteTurn, afterThreeMoves))

    def reset(self):
        self.__loadState(self.defaultGameState)

    def __init__(self, boardState = defaultGameState):
        self.__loadState(boardState)

    def __loadState(self, boardState):
        self.board = np.array([boardState[i] for i in range(24)])
        self.whiteForward = boardState[24].lower() == "f"
        self.blackForward = boardState[25].lower() == "f"
        self.whiteTurn = boardState[26].lower() == "w"
        self.afterThreeMoves = boardState[27] == "1"
    
    def __getAvailableMovesFromPos(self, pos):
        allowed = []
        if(pos < 0 or pos >= 24): return allowed
        piece = self.board[pos]
        if(piece == "-"): return allowed
        if(16 <= pos < 24):
            for index in range(16):
                if(self.board[index] == "-"):
                    allowed.append(index)
        elif(piece.lower() == "p"):
            if(piece.isupper()):
                direction = -1 if self.whiteForward else 1
            else:
                direction = 1 if self.blackForward else -1
            index = pos + direction * 4
            if(0 <=  index < 16 and self.board[index] == "-"):
                allowed.append(index)
            for index in [pos + direction * i for i in range(3, 6, 2)]:
                    if(0 <= index < 16 and 
                    abs((pos % 4) - (index % 4)) == 1 and 
                    (self.board[index] != "-" and 
                    self.board[index].isupper() != piece.isupper())):
                        allowed.append(index)
        elif(piece.lower() == "n"):
            directions = []
            col = pos % 4
            if(col < 2):
                directions += [-2, 6]
            else:
                directions += [2, -6]
            if(col > 0):
                directions += [7, -9]
            if(col < 3):
                directions += [-7, 9]
            for index in [pos + i for i in directions]:
                if(0 <= index < 16):
                    to = self.board[index]
                    if(to == "-" or to.isupper() != piece.isupper()):
                        allowed.append(index)
        elif(piece.lower() == "b" or piece.lower() == "r"):
            directions = [[3, 6, 9], [5, 10, 15]] if piece.lower() == "b" else [[4, 8, 12], [1, 2, 3]]
            for mul in range(-1, 2, 2):
                for direction in directions:
                    lastCol = pos % 4
                    for index in [pos + mul * i for i in direction]:
                        if(0 <= index < 16 and abs(lastCol - (index % 4)) == 1):
                            lastCol = index % 4
                            if(self.board[index] == "-"):
                                allowed.append(index)
                            else:
                                if(self.board[index].isupper() != piece.isupper()):
                                    allowed.append(index)
                                break
                        else: break
        return allowed

    @staticmethod
    def stateToString(board, whiteForward, blackForward, whiteTurn, afterThreeMoves):
        return "".join(np.append(board, [
            "f" if whiteForward else "b",
            "f" if blackForward else "b",
            "w" if whiteTurn else "b",
            "1" if afterThreeMoves else "0"
        ]))
