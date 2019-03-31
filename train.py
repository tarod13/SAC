import pickle
import numpy as np
from system import System
n_test = 6
system_type = 'Hopper-v2'
system = System(system=system_type, reward_scale=100)
tr_epsds = 200
epsd_steps = 1000
mean_rewards = []
for i in range(1,6):
    if i == 1:
        mean_rewards.append(system.train_agent(tr_epsds, epsd_steps))        
    else:
        mean_rewards.append(system.train_agent(tr_epsds, epsd_steps, initialization=False))
    pickle.dump(system,open(system_type+'_'+str(n_test)+'_'+str((i+5)//5)+'.p','wb'))
    np.savetxt('mean_rewards_'+system_type+'_'+str(n_test)+'.txt',mean_rewards)
