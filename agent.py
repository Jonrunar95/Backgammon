#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon
import torch
import pickle
import twolayernetog
import random
from IPython.core.debugger import set_trace



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def greedy(boards, model):

    x = torch.tensor(boards, dtype = torch.float, device = device)
    # now do a forward pass to evaluate the board's after-state value
    y = model(x)
    y_greedy, move = torch.max(y, 0)
    #chosenboard = torch.tensor(boards[move], dtype = torch.float, device = device)
    #y_greedy = model(chosenboard)

    return move, y_greedy


def learn(y_old, model, boards, winner):
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    if(winner == "yes" or winner == "no"):
        if(winner == "yes"):
            reward =  [1.0]
        elif (winner == "no"):
            reward =  [0.0]
        y_new = torch.tensor(reward, dtype = torch.float, device = device)
        real = y_new.data.cpu().numpy()[0]
        estimate = y_old.data.cpu().numpy()[0]
#        print("gameover", y_new.data.cpu().numpy()[0], y_old.data.cpu().numpy()[0], abs(real-estimate))
    else:
        move, y_new = greedy(boards, model)
    # now do a forward pass to evaluate the board's after-state value

    loss = criterion(y_old, y_new)
    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def dynaLearn(y_old, model, boards, dynaModel, winner):
#def dynaLearn(y_old, model, boards, winner):

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    if(winner == "yes" or winner == "no"):
        if(winner == "yes"):
            reward =  [1.0]
        elif (winner == "no"):
            reward =  [0.0]
        y_new = torch.tensor(reward, dtype = torch.float, device = device)
        real = y_new.data.cpu().numpy()[0]
        estimate = y_old.data.cpu().numpy()[0]
#        print("gameover", y_new.data.cpu().numpy()[0], y_old.data.cpu().numpy()[0], abs(real-estimate))
    else:
        move, y_new = greedy(boards, model)
    # now do a forward pass to evaluate the board's after-state value

    loss = criterion(y_old, y_new)
    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()
#    loss.backward()
    loss.backward(retain_graph=True)
    optimizer.step()

#    search(boards, n, dynaModel, model)
    search(5, dynaModel, model, y_old)   # 100?



#def search(n, dynaModel, model):
#    for i in range(n):
#        state_object = random.sample(dynaModel)
#        
#        state = state_object[0]
#        adterstate = state_object[1]
#        
#        print(state, afterstate)
#        
        

def search(n, dynaModel, model, y_old):

    for i in range(n):

        rnd = np.random.randint(len(dynaModel))
        state = dynaModel[rnd][0]
        afterstatestate = dynaModel[rnd][1]
        encoded_state = getinputboard(state)
        encoded_afterstate = getinputboard(afterstatestate)
        
#        # pick a random state s from the model
#        state_object = dynaModel[np.random.randint(len(dynaModel))]
#        state = state_object[0]
#        
#        # pick a random action (dice?) a from s
#        dice_object = state_object[1][np.random.randint(len(state_object[1]))]
#        dice = dice_object[0]
#        
#        action = dice_object[1][np.random.randint(len(dice_object[1]))]
        
        # R,newBoard/(state) <--- model(s,a)
        
        x_1 = torch.tensor(encoded_state, dtype = torch.float, device = device)
        y_1 = model(x_1)
        
        x_2 = torch.tensor(encoded_afterstate, dtype = torch.float, device = device)
        y_2 = model(x_2)

        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        loss = criterion(y_1, y_2)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # update neural network with R and newBoard
        


def action(board_copy,dice,player,i, y_old, model, firstMove, training):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)


   # if there are no moves available
    if len(possible_moves) == 0: 
        return [], y_old

    boards = []
    for board in possible_boards:
        boards.append(getinputboard(board))
    
    if(not firstMove and training):
        # learn
        learn(y_old, model, boards, "")
        


    # take greedy Action
    action, y_greedy = greedy(boards, model)
    move = possible_moves[action]
    
    # make the best move according to the policy
    return move, y_greedy




def dyna_action(board_copy,dice,player,i, y_old, model, firstMove, dynaModel, training):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)


   # if there are no moves available
    if len(possible_moves) == 0: 
        return [], y_old

    boards = []
    for board in possible_boards:
        boards.append(getinputboard(board))
    
    if(not firstMove and training):
        # learn
        dynaLearn(y_old, model, boards, dynaModel, "no")
#        dynaLearn(y_old, model, boards, "no")

    # take greedy Action
    action, y_greedy = greedy(boards, model)
    move = possible_moves[action]



    for m in move:
        afterstate = update_board(board_copy, m, player)


    # append or update the dynaModel
    update_dynaModel(dynaModel, board_copy, afterstate)
#    update_dynaModel(dynaModel, board_copy, dice, move)

    # make the best move according to the policy
    return move, y_greedy
    


def update_dynaModel(dynaModel, state, afterstate):
    
    state_missing = True

    state_object = (state, afterstate)

    for s in range(len(dynaModel)):
        if(state_object is dynaModel[s]):
            state_missing = False

#    for s in range(len(dynaModel)):
#        if(np.array_equal(state, dynaModel[s])):
#            state_missing = False
    
    if(state_missing):
        dynaModel.append(state_object)


#
## dynaModel:
## [
##    (S1, [ (d1, [m1, m2] ) ]),
##    (S2, [ (d1, [m1]), (d2, [m1, m2, m3]) ]),
##    (S3, [ etc. ]) ...
## ]
#def update_dynaModel(dynaModel, board_copy, dice, move):
#    
#    state_missing = True
#    dice_missing = True
#    move_missing = True
#
#    # dice are sorted to prevent duplicates
#    dice = np.sort(dice)
#
#    for s in range(len(dynaModel)):
#        if(np.array_equal(board_copy, dynaModel[s][0])):
#            state_missing = False
#
#            for d in range(len(dynaModel[s][1])):
#                if(np.array_equal(dice, dynaModel[s][1][d][0])):
#                    dice_missing = False
#                    
#                    for m in range(len(dynaModel[s][1][d][1])):
#                        if(np.array_equal(move, dynaModel[s][1][d][1][m])):
#                            move_missing = False
#
#                    if(move_missing):
#                        dynaModel[s][1][d][1].append(move)
#
#            if(dice_missing):
#                dice_object = (dice, [move])
#                dynaModel[s][1].append(dice_object)
#
#    if(state_missing):
#        state_object = (board_copy, [ (dice, [move]) ])
#        dynaModel.append(state_object)



def getinputboard(board):
    boardencoding = np.zeros(15*28*2)
    for i in range(1, len(board)):
        val = board[i]
        if(val > 0):
            boardencoding[(i-1)*15 + int(board[i])] = 1
        elif(val < 0):
            boardencoding[(i-1)*15 + int(abs(board[i])) + 360] = 1         
    return boardencoding


def update_board(board, move, player):
    # updates the board
    # inputs are some board, one move and the player
    # outputs the updated board
    board_to_update = np.copy(board) 

    # if the move is there
    if len(move) > 0:
        startPip = move[0]
        endPip = move[1]
        
        # moving the dead piece if the move kills a piece
        kill = board_to_update[endPip]==(-1*player)
        if kill:
            board_to_update[endPip] = 0
            jail = 25+(player==1)
            board_to_update[jail] = board_to_update[jail] - player
        
        board_to_update[startPip] = board_to_update[startPip]-1*player
        board_to_update[endPip] = board_to_update[endPip]+player

    return board_to_update