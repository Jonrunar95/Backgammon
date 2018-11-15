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



#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

def greedy(boards, model):

    x = torch.tensor(boards, dtype = torch.float, device = device)
    # now do a forward pass to evaluate the board's after-state value
    #model = model.cuda()
    y = model(x)
    y_greedy, move = torch.max(y, 0)
    #chosenboard = torch.tensor(boards[move], dtype = torch.float, device = device)
    #y_greedy = model(chosenboard)

    return move, y_greedy

def learn(y_old, model, board, player, gameOver):
    #model = model.cuda()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    if(gameOver):
        if(player == -1):
            reward =  [1.0]
        else:
            reward =  [0.0]
        y_new = torch.tensor(reward, dtype = torch.float, device = device)
    else:
        x = torch.tensor(board, dtype = torch.float, device = device)
        y_new = model(x)
    # now do a forward pass to evaluate the board's after-state value
   
    if(gameOver):
        print("yo", reward, y_new, player)
    loss = criterion(y_old, y_new)
    # Zero gradients, perform a backward pass, and update the weights.

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    

def action(board_copy,dice,player,i, y_greedy, model):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [], y_greedy

    boards = []
    for board in possible_boards:
        boards.append(getinputboard(board))
    action, y_greedy = greedy(boards, model)
    move = possible_moves[action]

    # update the board
    learning_board = np.copy(board_copy)
    if len(move) != 0:
        for m in move:
            learning_board = Backgammon.update_board(learning_board, m, player)

    
    learning_board = getinputboard(learning_board)
    learn(y_greedy, model, learning_board, player, False)
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
