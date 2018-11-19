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
from IPython.core.debugger import set_trace



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

def greedy(boards, model):

    x = torch.tensor(boards, dtype = torch.float, device = device)
    # now do a forward pass to evaluate the board's after-state value
    y = model(x)
    y_greedy, move = torch.max(y, 0)
    #chosenboard = torch.tensor(boards[move], dtype = torch.float, device = device)
    #y_greedy = model(chosenboard)

    return move, y_greedy

def learn(y_old, model, boards, player, gameOver):
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    if(gameOver):
        if(player == -1):
            reward =  [1.0]
        else:
            reward =  [0.0]
        y_new = torch.tensor(reward, dtype = torch.float, device = device)
        #print("gameover", y_new.data.cpu().numpy()[0], y_old.data.cpu().numpy()[0])
    else:
        move, y_new = greedy(boards, model)
    # now do a forward pass to evaluate the board's after-state value

    loss = criterion(y_old, y_new)
    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    

def action(board_copy,dice,player,i, y_old, model, firstMove):
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
    
    if(not firstMove):
        # learn
        learn(y_old, model, boards, player, False)
        


    # take greedy Action
    action, y_greedy = greedy(boards, model)
    move = possible_moves[action]
    
    # make the best move according to the policy
    return move, y_greedy

def getinputboard(board):
    boardencoding = np.zeros(15*28*2)
    for i in range(1, len(board)):
        val = board[i]
        if(val > 0):
            boardencoding[(i-1)*15 + int(board[i])] = 1
        elif(val < 0):
            boardencoding[(i-1)*15 + int(abs(board[i])) + 360] = 1         
    return boardencoding
