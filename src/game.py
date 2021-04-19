import constants
import numpy as np

class Game:

    defaultGameState = "---------------P-NBRrbnpffw0"

    def getWinner(self): # 1 for white, -1 for black, 0 for no winner
        for win in constants.wins:
            if("-" in [self.board[i] for i in win]):
                return 0
            winner = constants.pieces[self.board[win[0]]]["color"]
            if([constants.pieces[self.board[i]]["color"] for i in win].count(winner) == 4):
                return  1 if winner == "white" else -1
        return 0

    def getAvailableMoves(self, pos): # TODO: Update this to do it for every position on the board
        allowed = []
        if(pos < 0 or pos >= 24): return allowed
        piece = self.board[pos]
        if(piece == "-"): return allowed
        if(16 <= pos < 24):
            for index in range(16):
                if(self.board[index] == "-"):
                    allowed.append(index)
        elif(piece.lower() == "p"): # pawn
            if(piece.isupper()):
                direction = -1 if self.whiteForward else 1
            else:
                direction = 1 if self.blackForward else -1
            index = pos + direction * 4
            if(0 <=  index < 16 and self.board[index] == "-"):
                allowed.append(index)
            for index in [pos + direction * i for i in range(3, 6, 2)]:
                if(0 <= index < 16 and self.board[index] != "-" and self.board[index].isupper() != piece.isupper()):
                    allowed.append(index)
        elif(piece.lower() == "n"): # knight
            directions = [-2, -6, -7, -9, 2, 6, 7, 9]
            for index in [pos + i for i in directions]:
                if(0 <= index < 16):
                    to = self.board[index]
                    if(to == "-" or to.isupper() != piece.isUpper()):
                        allowed.append(index)
        else: # bishop and rook (since logic is similar)
            directions = [[3, 6, 9], [5, 10, 15]] if piece.lower() == "b" else [[4, 8, 12], [1, 2, 3]]
            for mul in range(-1, 2, 2):
                for direction in directions:
                    for index in [pos + mul * i for i in direction]:
                        if(0 <= index < 16):
                            if(self.board[index] == "-"):
                                allowed.append(index)
                            else:
                                if(self.board[index].isupper() != piece.isupper()):
                                    allowed.append(index)
                                break
        return allowed

    def toString(self):
        return Game.stateToString(self.board, self.whiteForward, self.blackForward, self.whiteTurn, self.afterThreeMoves)

    def toModelInput(self):
        modelInput = []
        # piece planes
        for piece in constants.pieces:
            plane = np.zeros(24, dtype=np.int)
            plane[self.board==piece] = 1
            modelInput = np.append(modelInput, plane)
        # white pawn direction
        modelInput = np.append(modelInput, np.ones(24, dtype=np.int) if self.whiteForward else np.zeros(24, dtype=np.int))
        # black pawn direction
        modelInput = np.append(modelInput, np.ones(24, dtype=np.int) if self.blackForward else np.zeros(24, dtype=np.int))
        # after three moves
        modelInput = np.append(modelInput, np.ones(24, dtype=np.int) if self.afterThreeMoves else np.zeros(24, dtype=np.int))
        return np.array([np.reshape(modelInput, (11, 4, 6))])
    
    # Note: This method does not check if an action is legal
    def takeAction(self, action): # action is (from, to)
        newBoard = np.array(self.board)
        to = newBoard[action[1]]
        piece = newBoard[action[0]]
        if(to != "-"): # Replace piece to offboard
            newBoard[16 + list(constants.pieces).index(to)] = to
        newBoard[action[1]] = piece
        newBoard[action[0]] = "-"
        whiteForward = self.whiteForward
        blackForward = self.blackForward
        if(piece.lower() == "p" and action[1] in [0, 1, 2, 3, 12, 13, 14, 15]):
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

    @staticmethod
    def stateToString(board, whiteForward, blackForward, whiteTurn, afterThreeMoves):
        return "".join(np.append(board, [
            "f" if whiteForward else "b",
            "f" if blackForward else "b",
            "w" if whiteTurn else "b",
            "1" if afterThreeMoves else "0"
        ]))
