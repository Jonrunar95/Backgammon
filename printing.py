import numpy as np



def moves_to_string(moves):
    for i in range(len(moves)):
        print("Move", i, ":", end = " ")
        for j in range(len(moves[i])):
            print(moves[i][j], end = " ")
        print()