import pickle
import numpy as np
from system import System
pendulum = System()
tr_epsds = 80
epsd_steps = 500
mean_rewards = []
for i in range(1,2):
    if i == 1:
        mean_rewards.append(pendulum.train_agent(tr_epsds, epsd_steps))
        pickle.dump(pendulum,open('pendulum_9_1.p','wb'))
        np.savetxt('mean_rewards_pendulum_9.txt',mean_rewards)
    else:
        mean_rewards.append(pendulum.train_agent(tr_epsds, epsd_steps, initialization=False))
        pickle.dump(pendulum,open('pendulum_9_'+str(i)+'.p','wb'))
        np.savetxt('mean_rewards_pendulum_9.txt',mean_rewards)
