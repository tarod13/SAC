# from system import System
# pendulum = System()
# tr_epsds = 125
# epsd_steps = 800
# mean_rewards = []
# for i in range(0,10):
#     if i == 0:
#         mean_rewards.append(pendulum.train_agent(tr_epsds, epsd_steps))
#     else:
#         mean_rewards.append(pendulum.train_agent(tr_epsds, epsd_steps, initialization=False))

# import time
# import cProfile
# from system import System

# def main():
#     hopper = System()
#     tr_epsds = 50
#     epsd_steps = 800
#     mean_rewards = []
#     mean_rewards.append(hopper.train_agent(tr_epsds, epsd_steps))
#     # for i in range(0,1):
#     #     if i == 0:
#     #         mean_rewards.append(hopper.train_agent(tr_epsds, epsd_steps))
#     #     else:
#     #         mean_rewards.append(hopper.train_agent(tr_epsds, epsd_steps, initialization=False))

#     import pickle
#     pickle.dump(hopper,open('hopper_3.p','wb'))
#     import numpy as np
#     np.savetxt('mean_rewards_3".txt',mean_rewards)

# if __name__ == '__main__':
#     main()
#     # cProfile.run('main()')

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
