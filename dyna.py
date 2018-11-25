import numpy as np
    
    model = np.zeros(states, actions, 2)
    
    S = newGame()
    while(not terminal):
        A = egreedy(S, Q)
        R, S_, terminal = step(S, A)
        Q[S][A] += alpha * (R + gamma * np.max(Q[S_]) - Q[S][A])
        model[S][A] = (R, S_)
        for i in range(n):
            randomS = randomState(model)
            randomA = randomAction(model[S])
            R, randomS_ = model[randomS][randomA]
            Q[randomS][randomA] += alpha * (R + gamma * np.max(Q[randomS_]) - Q[randomS][randomA])
        S = S_