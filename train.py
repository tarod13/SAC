import pickle
import numpy as np
from system import System
hopper = System()
tr_epsds = 200
epsd_steps = 1000
mean_rewards = []
for i in range(1,6):
    if i == 1:
        mean_rewards.append(hopper.train_agent(tr_epsds, epsd_steps))
        pickle.dump(hopper,open('hopper_3_1.p','wb'))
        np.savetxt('mean_rewards_hopper_3.txt',mean_rewards)
    else:
        mean_rewards.append(hopper.train_agent(tr_epsds, epsd_steps, initialization=False))
        pickle.dump(hopper,open('hopper_3_'+str(i)+'.p','wb'))
        np.savetxt('mean_rewards_hopper_3.txt',mean_rewards)
