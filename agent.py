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


device = torch.device("cpu")
firstMove1 = True
y_new = 0
y_old = 0

def greedy(board, w1, b1, w2, b2):

    # encode the board to create the input
    board = board[1:]
    print(len(board[0]), len(board))
    # https://pytorch.org/docs/stable/autograd.html#variable-deprecated
    x = Variable(torch.tensor(board, dtype = torch.float, device = device)).view(len(board[0]), len(board))
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_relu= h.clamp(min=0) # squash this with a sigmoid function
    y_pred = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    va = y.clamp(min=0)
    y_new = max(va)
    return np.argmax(va), y_new

def learn(y_old, w1, b1, w2, b2, board):
    x = Variable(torch.tensor(board, dtype = torch.float, device = device)).view(28,1)
    # now do a forward pass to evaluate the board's after-state value
    h = torch.mm(w1,x) + b1 # matrix-multiply x with input weight w1 and add bias
    h_relu= h.clamp(min=0) # squash this with a sigmoid function
    y_pred = torch.mm(w2,h_sigmoid) + b2 # multiply with the output weights w2 and add bias
    va = y.clamp(min=0)
    y_new = max(va)

    delta = y_new-y_old
    y_old.backward()

    w1.data += 0.01*delta*w1.grad.data
    w2.data += 0.01*delta*w2.grad.data
    b1.data += 0.01*delta*b1.grad.data
    b2.data += 0.01*delta*b2.grad.data
    
    #error = o - y
    #o_delta = sigmoid_derivative(o)*error
    #z2_error = o_delta.dot(w2.T)
    #h_sigmoid().backward()
    #z2_delta = z2_error * sigmoid_derivative(h_sigmoid)
    #w1 += x.T.dot(z2_delta)
    #w2 = h_sigmoid * delta


def action(board_copy,dice,player,i):
    if firstMove1 == True:
        
        w1 = torch.randn(28*28, 28, device = device, dtype=torch.float, requires_grad = True)
        b1 = torch.zeros((28*28,1), device = device, dtype=torch.float, requires_grad = True)
        w2 = torch.randn(1,28*28, device = device, dtype=torch.float, requires_grad = True)
        b2 = torch.zeros((1,1), device = device, dtype=torch.float, requires_grad = True)
        firstMove1==False
    else:
        y_old = y_new
        learn(y_old, w1, b1, w2, b2)
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return []

    action, y_new = greedy(possible_boards, w1, b1, w2, b2)
    # make the best move according to the policy


    # policy missing, returns a random move for the time being
    move = possible_moves[action]

    return move

def getinputboard(board):

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

    return boardencoding
