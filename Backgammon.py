#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backgammon interface
Run this program to play a game of Backgammon
The agent is stored in another file 
Most (if not all) of your agent-develeping code should be written in the agent.py file
Feel free to change this file as you wish but you will only submit your agent 
so make sure your changes here won't affect his performance.
"""
import numpy as np
import agent
import torch
import time
import pickle
import twolayernetog
import printing

import flipped_agent 
device = torch.device("cuda")
def init_board():
    # initializes the game board
    board = np.zeros(29)
    board[1] = -2
    board[12] = -5
    board[17] = -3
    board[19] = -5
    board[6] = 5
    board[8] = 3
    board[13] = 5
    board[24] = 2
    return board

def roll_dice():
    # rolls the dice
    dice = np.random.randint(1,7,2)
    return dice

def game_over(board):
    # returns True if the game is over    
    return board[27]==15 or board[28]==-15

def check_for_error(board):
    # checks for obvious errors
    errorInProgram = False
    
    if (sum(board[board>0]) != 15 or sum(board[board<0]) != -15):
        # too many or too few pieces on board
        errorInProgram = True
        print("Too many or too few pieces on board!", sum(board[board>0]), sum(board[board<0]))
    return errorInProgram
    
def pretty_print(board):
    string = str(np.array2string(board[1:13])+'\n'+
                 np.array2string(board[24:12:-1])+'\n'+
                 np.array2string(board[25:29]))
    print("board: \n", string)
    
            
        
def legal_move(board, die, player):
    # finds legal moves for a board and one dice
    # inputs are some BG-board, the number on the die and which player is up
    # outputs all the moves (just for the one die)
    possible_moves = []

    if player == 1:
        
        # dead piece, needs to be brought back to life
        if board[25] > 0: 
            start_pip = 25-die
            if board[start_pip] > -2:
                possible_moves.append(np.array([25,start_pip]))
                
        # no dead pieces        
        else:
            # adding options if player is bearing off
            if sum(board[7:25]>0) == 0: 
                if (board[die] > 0):
                    possible_moves.append(np.array([die,27]))
                    
                elif not game_over(board): # smá fix
                    # everybody's past the dice throw?
                    s = np.max(np.where(board[1:7]>0)[0]+1)
                    if s<die:
                        possible_moves.append(np.array([s,27]))
                    
            possible_start_pips = np.where(board[0:25]>0)[0]

            # finding all other legal options
            for s in possible_start_pips:
                end_pip = s-die
                if end_pip > 0:
                    if board[end_pip] > -2:
                        possible_moves.append(np.array([s,end_pip]))
                        
    elif player == -1:
        # dead piece, needs to be brought back to life
        if board[26] < 0: 
            start_pip = die
            if board[start_pip] < 2:
                possible_moves.append(np.array([26,start_pip]))
                
        # no dead pieces       
        else:
            # adding options if player is bearing off
            if sum(board[1:19]<0) == 0: 
                if (board[25-die] < 0):
                    possible_moves.append(np.array([25-die,28]))
                elif not game_over(board): # smá fix
                    # everybody's past the dice throw?
                    s = np.min(np.where(board[19:25]<0)[0])
                    if (6-s)<die:
                        possible_moves.append(np.array([19+s,28]))

            # finding all other legal options
            possible_start_pips = np.where(board[0:25]<0)[0]
            for s in possible_start_pips:
                end_pip = s+die
                if end_pip < 25:
                    if board[end_pip] < 2:
                        possible_moves.append(np.array([s,end_pip]))
        
    return possible_moves

def legal_moves(board, dice, player):
    # finds all possible moves and the possible board after-states
    # inputs are the BG-board, the dices rolled and which player is up
    # outputs the possible pair of moves (if they exists) and their after-states

    moves = []
    boards = []

    # try using the first dice, then the second dice
    possible_first_moves = legal_move(board, dice[0], player)
    for m1 in possible_first_moves:
        temp_board = update_board(board,m1,player)
        possible_second_moves = legal_move(temp_board,dice[1], player)
        for m2 in possible_second_moves:
            moves.append(np.array([m1,m2]))
            boards.append(update_board(temp_board,m2,player))
        
    if dice[0] != dice[1]:
        # try using the second dice, then the first one
        possible_first_moves = legal_move(board, dice[1], player)
        for m1 in possible_first_moves:
            temp_board = update_board(board,m1,player)
            possible_second_moves = legal_move(temp_board,dice[0], player)
            for m2 in possible_second_moves:
                moves.append(np.array([m1,m2]))
                boards.append(update_board(temp_board,m2,player))
            
    # if there's no pair of moves available, allow one move:
    if len(moves)==0: 
        # first dice:
        possible_first_moves = legal_move(board, dice[0], player)
        for m in possible_first_moves:
            moves.append(np.array([m]))
            boards.append(update_board(temp_board,m,player))
            
        # second dice:
        possible_first_moves = legal_move(board, dice[1], player)
        for m in possible_first_moves:
            moves.append(np.array([m]))
            boards.append(update_board(temp_board,m,player))
            
    return moves, boards 

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
    
    
def random_agent(board_copy,dice,player,i):
    # random agent
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move randomly
    
    # check out the legal moves available for dice throw
    possible_moves, possible_boards = legal_moves(board_copy, dice, player)
    
    # if there are no possible moves, return empty move:
    if len(possible_moves) == 0: 
        return []
    
    # pick a random move:
    move = possible_moves[np.random.randint(len(possible_moves))]
    return move
    

def play_a_game(opponent, commentary = False):
    board = init_board() # initialize the board
    player = np.random.randint(2)*2-1 # which player begins?
    y_old = 0
    y_old_p2 = 0
    firstMove = True
    firstMove_p2 = True
    pickle_in = open("randommodel.pickle","rb")
    model = pickle.load(pickle_in)
    model = model.cuda()
    # play on
    while not game_over(board) and not check_for_error(board):
        if commentary: print("lets go player ",player)
        
        # roll dice
        dice = roll_dice()
        if commentary: print("rolled dices:", dice)
            
        # make a move (2 moves if the same number appears on the dice)
        for i in range(1+int(dice[0] == dice[1])):
            board_copy = np.copy(board) 

            # make the move (agent vs agent):
            if(opponent == "agent"):
                if player == 1:
                    move, y_old = agent.action(board_copy,dice,player,i, y_old, model, firstMove, True)
                    # update the board
                    if len(move) != 0:
                        for m in move:
                            board = update_board(board, m, player)
                    if(firstMove):
                        firstMove = False
                elif player == -1:
                    flipped_board = flipped_agent.flip_board(board_copy)
                    move, y_old_p2 = agent.action(flipped_board,dice,1,i, y_old_p2, model, firstMove_p2, True)
                    if len(move) != 0:
                        for m in move:
                            flipped_board = update_board(flipped_board, m, 1)
                    board = flipped_agent.flip_board(flipped_board)
            
                    if(firstMove_p2):
                        firstMove_p2 = False
            elif(opponent == "human"):
                pretty_print(board)
                if player == 1:
                    print("Computer's turn")
                    move, y_old = agent.action(board_copy,dice,player,i, y_old, model, firstMove, False)
                    print("Computer's move", move)
                elif player == -1:
                    print("Human's turn")
                    possible_moves, possible_boards = legal_moves(board_copy, dice, player)
                    print("dice:", dice)
                    printing.moves_to_string(possible_moves)
                    text = input("prompt")
                    move = possible_moves[int(text)]

                if len(move) != 0:
                    for m in move:
                        board = update_board(board, m, player)
            #if you're playing vs random agent:
            elif(opponent == "random"):
                if player == 1:
                    move, y_old = agent.action(board_copy,dice,player,i, y_old, model, firstMove, False)
                elif player == -1:
                    move = random_agent(board_copy,dice,player,i)
                if len(move) != 0:
                        for m in move:
                            board = update_board(board, m, player)
            # update the board

            
            # give status after every move:         
            if commentary: 
                print("move from player",player,":")
                pretty_print(board)
        

        # players take turns 
        player = -player


    # return the winner
    winner = -1*player
    if(opponent == "agent"):
        if(winner == 1):
            agent.learn(y_old, model, board_copy, "yes")
            agent.learn(y_old_p2, model, board_copy, "no")
        else:
            agent.learn(y_old, model, board_copy, "no")
            agent.learn(y_old_p2, model, board_copy, "yes")
        
    #print("Winner is player", winner)
    pickle_out = open("randommodel.pickle","wb")
    pickle.dump(model, pickle_out)
    pickle_out.close()
    return winner
    

def main():
    winners = {}; winners["1"]=0; winners["-1"]=0; # Collecting stats of the games
    nGames = 0 # how many games?
    starttime = time.time()
    for g in range(nGames):
        if(g%(nGames/100) == 0):
            percent = int(g/(nGames/100))
            timenow = time.time()
            timediff = timenow - starttime
            if(g != 0):
                totaltime = int((timediff/(g/nGames))/60) 
                print("Games played:", g, "--- Done", percent,"%",
                "Time done:", int(timediff/60), "minutes --- Total time:", totaltime, "minutes")
        winner = play_a_game("agent", commentary=False)
        winners[str(winner)] += 1
    print("Out of", nGames, "games,"),0
    print("player", 1, "won", winners["1"],"times and")
    print("player", -1, "won", winners["-1"],"times")
    games = 100
    winners = {}; winners["1"]=0; winners["-1"]=0; # Collecting stats of the games
    for g in range(games):
        winner = play_a_game("random", commentary=False)
        winners[str(winner)] += 1
    print("Out of", games, "games,"),0
    print("player", 1, "won", winners["1"],"times and")
    print("player", -1, "won", winners["-1"],"times")

if __name__ == '__main__':
    main()
    
    

