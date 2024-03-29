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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def greedy(boards, model):

    x = torch.tensor(boards, dtype = torch.float, device = device)
    # now do a forward pass to evaluate the board's after-state value
    y = model(x)
    y_greedy, action = torch.max(y, 0)

    return action, y_greedy

def learn(y_old, model, boards, winner):
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    action = 0
    if(winner == "yes" or winner == "no"):
        if(winner == "yes"):
            reward =  [1.0]
        elif (winner == "no"):
            reward =  [0.0]
        y_new = torch.tensor(reward, dtype = torch.float, device = device)
    else:
        action, y_new = greedy(boards, model)
    # now do a forward pass to evaluate the board's after-state value
    loss = criterion(y_old, y_new)
    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    

def action(board_copy,dice,player,i, y_old, model, firstMove, training): 
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
    action, y_new = greedy(boards, model)  
    move = possible_moves[action]  
    # make the best move according to the policy
    return move, y_new

def getinputboard(board):
    boardencoding = np.zeros(15*28*2)
    for i in range(1, len(board)):
        val = board[i]
        if(val > 0):
            boardencoding[(i-1)*15 + int(board[i])] = 1
        elif(val < 0):
            boardencoding[(i-1)*15 + int(abs(board[i])) + 360] = 1         
    return boardencoding
