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
from torch.autograd import Variable

def sigmoid(x):
    return 1.0/(1+ np.exp(-x)),

def sigmoid_derivative(x):
    return x * (1.0 - x)

def greedy(board, w1, b1, w2, b2):
    na = np.size(board)
    va = np.zeros(na)
    for i in range(0, na):
        # encode the board to create the input
        nn = board*(1/15)

        # https://pytorch.org/docs/stable/autograd.html#variable-deprecated
        x = Variable(torch.tensor(nn, dtype = torch.float, device = device)).view(28,1)
        # now do a forward pass to evaluate the board's after-state value
        h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
        h_sigmoid = h.sigmoid() # squash this with a sigmoid function
        y = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
        va[i] = y.sigmoid()
    
    return np.argmax(va)

def learn(w1, b1, w2, b2):
    print(w1, b1, w2, b2)



def action(board_copy,dice,player,i):
    if firstMove:
        w1 = Variable(torch.randn(28*28, 28, device = device, dtype=torch.float), requires_grad = True)
        b1 = Variable(torch.zeros((28*28,1), device = device, dtype=torch.float), requires_grad = True)
        w2 = Variable(torch.randn(1,28*28, device = device, dtype=torch.float), requires_grad = True)
        b2 = Variable(torch.zeros((1,1), device = device, dtype=torch.float), requires_grad = True)
    else:
        learn(w1, b1, w2, b2)
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return []

    action = greedy(possible_boards, w1, b1, w2, b2)
    # make the best move according to the policy


    # policy missing, returns a random move for the time being
    move = possible_moves[action]

    return move

# gitkraken test