
import numpy as np

board = np.zeros(29)
board[1] = -2
#board[1] = 2
board[12] = -5
board[17] = -3
board[19] = -5
board[6] = 5
board[8] = 3
board[13] = 5
board[24] = 2

def getinputboard(board):
    # v1
    # each position, jail and finish
    # [010 ... 0] x 24 x 2, [00][00] ?

    boardencoding = np.zeros(15*24*2 + 4)

    for i in range(1, 25):
#        print(str(i) + " " + str(board[i]))
        if board[i] > 0:
            boardencoding[(i-1)*15 + int(board[i])] = 1
#            print("white")
        elif board[i] < 0:
            boardencoding[(i-1)*15 + int(abs(board[i])) + 360] = 1
#            print("black")
            
    boardencoding[720] = board[25]
    boardencoding[721] = board[26]
    boardencoding[722] = board[27]
    boardencoding[723] = board[28]


# PUT INTO THE FREEZER    
#
# v2
# [1000] = 0 disks, [1100] = 1 d, [1110] = 2 d, [1111] = 3 d, [111(4-2)] = [1112] = 4 d
# [0100] x 24 x 2 + [00][00] = 96 + 96 + 4 = 196
#
#    boardencoding = np.zeros(4*24*2 + 4)
#    
#    for i in range(1, len(board)):
#        print(str(i) + " " + str(board[i]))
#        
#        if board[i] == 0:
#            
#            print("zero")
#        elif board[i] > 0:
#            if board[i] >= 1:
#                boardencoding[(i-1) * 4 + 1] = 1
#            if board[i] >= 2:
#                boardencoding[(i-1) * 4 + 2] = 1
#            if board[i] > 2:
#                boardencoding[(i-1) * 4 + 3] = abs(board[i])
#            
#            print("white " + str(boardencoding[i]))
#            
#        else:
#            if board[i] == -1:
#                boardencoding[i * 4 + 96] = 1
#            if board[i] == -2:
#                boardencoding[i * 4 + 97] = 1
#            if board[i] < -2:
#                boardencoding[i * 4 + 98] = abs(board[i])
#            
#            print("black " + str(boardencoding[i+96]))
            
            
        

    
    return boardencoding

test = getinputboard(board)